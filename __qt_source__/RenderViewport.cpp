#include "RenderViewport.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <chrono>
#include <QFileDialog>
#include <QIntValidator>
#include <QStyle>


RenderViewport::RenderViewport(QWidget *parent, const char *filename) : QWidget(parent), window(updateImagePixmap, this), instance(&context, &window) {
	ui.setupUi(this);
	flags = 0;

	// RESOLUTION:
	window.setWidth(1920);
	window.setHeight(1080);
	imageScale = 1.0f;
	resolutionSettings = RESOLUTION_FIXED_RELATIVE_SCALE;
	ui.widthInput->setValidator(new QIntValidator(0, 32000, this));
	ui.heightInput->setValidator(new QIntValidator(0, 32000, this));
	ui.widthInput->setText("1920");
	ui.heightInput->setText("1080");
	connect(ui.widthInput, SIGNAL(textChanged(const QString &)), this, SLOT(setWidth(const QString &)));
	connect(ui.heightInput, SIGNAL(textChanged(const QString &)), this, SLOT(setHeight(const QString &)));

	// IMAGE UPDATE SIGNALS:
	connect(this, SIGNAL(updateImage()), this, SLOT(imageUpdated()));
	connect(this, SIGNAL(issueSizeFix()), this, SLOT(fixImageSize()));
	connect(this, SIGNAL(issueSetStatusText(const QString &)), this, SLOT(setStatusText(const QString &)));

	// STATE BUTTONS:
	enableButtons(false, false, false);
	ui.startButton->setIcon(ui.startButton->style()->standardIcon(QStyle::SP_MediaPlay));
	ui.stopButton->setIcon(ui.stopButton->style()->standardIcon(QStyle::SP_MediaPause));
	ui.restartButton->setIcon(ui.restartButton->style()->standardIcon(QStyle::SP_MediaSeekBackward));
	connect(ui.startButton, SIGNAL(clicked()), this, SLOT(startRender()));
	connect(ui.stopButton, SIGNAL(clicked()), this, SLOT(stopRender()));
	connect(ui.restartButton, SIGNAL(clicked()), this, SLOT(restartRender()));
	connect(this, SIGNAL(issueEnableButtons(bool, bool, bool)), this, SLOT(enableButtons(bool, bool, bool)));

	// ITERATION INFO:
	refreshIterationInfo(0, 0);
	instance.onIterationComplete(iterationEndedCallback, this);
	connect(this, SIGNAL(issueRefreshIterationInfo(int, double)), this, SLOT(refreshIterationInfo(int, double)));

	// SAVE/LOAD:
	ui.saveButton->setIcon(ui.saveButton->style()->standardIcon(QStyle::SP_DialogSaveButton));
	ui.loadButton->setIcon(ui.loadButton->style()->standardIcon(QStyle::SP_DialogOpenButton));
	connect(ui.saveButton, SIGNAL(clicked()), this, SLOT(saveImage()));
	connect(ui.loadButton, SIGNAL(clicked()), this, SLOT(requestLoadScene()));

	// ZOOM/SCALE/ETC:
	ui.fitButton->hide();
	connect(ui.zoomButton, SIGNAL(clicked()), this, SLOT(zoomPressed()));
	connect(ui.fitButton, SIGNAL(clicked()), this, SLOT(fitPressed()));
	connect(ui.zoomSlider, SIGNAL(valueChanged(int)), this, SLOT(zoomChanged(int)));

	// RESOURCE SETTINGS:
	for (int i = 0; i <= std::thread::hardware_concurrency(); i++)
		ui.cpuThreadCount->addItem(((std::stringstream*)(&(std::stringstream() << i)))->str().c_str());
	ui.cpuThreadCount->setCurrentIndex(std::thread::hardware_concurrency());
	connect(ui.cpuThreadCount, SIGNAL(currentIndexChanged(int)), this, SLOT(cpuThreadCountChanged(int)));

	// COMMAND THREAD:
	commandThread = std::thread(controlThread, this);
	if (filename != NULL) {
		sceneFile = filename;
		emit sourceFileChanged(sceneFile);
		dispatchCommand(COMMAND_LOAD_SCENE);
	}
	updateThread = std::thread(maintenanceThread, this);
}

RenderViewport::~RenderViewport() {
	instance.interruptRender();
	instance.stop();
	dispatchCommand(COMMAND_QUIT);
	flags |= STATE_DESTRUCTOR_CALLED;
	commandThread.join();
	updateThread.join();
}


void RenderViewport::onIterationEnded() {
	emit issueRefreshIterationInfo(instance.iteration(), instance.renderTime());
}
void RenderViewport::iterationEndedCallback(void *viewport) {
	((RenderViewport*)viewport)->onIterationEnded();
}

void RenderViewport::fixImageSize() {
	float viewW, viewH, rendW, rendH, imW, imH, imX, imY;
	{
		QSize size = ui.scrollArea->size();
		viewW = ((float)size.width());
		viewH = ((float)size.height());
	}
	{
		int width, height;
		window.getResolution(width, height);
		rendW = ((float)width);
		rendH = ((float)height);
	}
	if (resolutionSettings == RESOLUTION_MATCH_VIEWPORT) {
		window.setImageResolution((int)viewW, (int)viewH);
		imW = viewW;
		imH = viewH;
	}
	else if (resolutionSettings == RESOLUTION_FIXED_CONSTANT_SCALE) {
		imW = (imageScale * rendW);
		imH = (imageScale * rendH);
	}
	else if (resolutionSettings == RESOLUTION_FIXED_RELATIVE_SCALE) {
		float scaleFactor;
		if ((viewW / viewH) > (rendW / rendH)) scaleFactor = (viewH / rendH);
		else scaleFactor = (viewW / rendW);
		imW = (imageScale * rendW * scaleFactor);
		imH = (imageScale * rendH * scaleFactor);
	}
	if (imW >= viewW) imX = 0;
	else imX = ((viewW - imW) * 0.5f);
	if (imH >= viewH) imY = 0;
	else imY = ((viewH - imH) * 0.5f);
	ui.image->resize((int)imW, (int)imH);
	ui.image->move((int)imX, (int)imY);
	if (resolutionSettings == RESOLUTION_MATCH_VIEWPORT || (resolutionSettings == RESOLUTION_FIXED_RELATIVE_SCALE && imageScale <= 1.0f))
		ui.scrollAreaContent->setMinimumSize(0, 0);
	else ui.scrollAreaContent->setMinimumSize((int)imW, (int)imH);
}

void RenderViewport::zoomPressed() {
	ui.zoomButton->hide();
	ui.fitButton->show();
	resolutionSettings = RESOLUTION_FIXED_CONSTANT_SCALE;
	ui.zoomSlider->setValue(0);
	imageScale = 1.0f;
	fixImageSize();
}
void RenderViewport::fitPressed() {
	ui.fitButton->hide();
	ui.zoomButton->show();
	resolutionSettings = RESOLUTION_FIXED_RELATIVE_SCALE;
	ui.zoomSlider->setValue(0);
	imageScale = 1.0f;
	fixImageSize();
}

void RenderViewport::zoomChanged(int sliderValue) {
	imageScale = ((sliderValue <= 0) ?
		(1.0f - (((float)-sliderValue) / ((float)-ui.zoomSlider->minimum()))) :
		(1.0f + (3.0f * pow(((float)sliderValue) / ((float)ui.zoomSlider->maximum()), 4.0f))));
	fixImageSize();
}

void RenderViewport::saveImage() {
	QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("PNG (*.png);;JPG (*.jpg);;TIFF (*.tiff)"));
	if (filename.length() <= 0) return;
	QFile file(filename);
	file.open(QIODevice::WriteOnly);
	pixmapLock.lock();
	pixmap.save(&file);
	pixmapLock.unlock();
}

void RenderViewport::requestLoadScene() {
	QString filename = QFileDialog::getOpenFileName(this, tr("Open File"), "", tr("DUMB (*.dumb);;ALL (*)"));
	if (filename.length() <= 0) return;
	sceneFile = filename.toUtf8().constData();
	emit sourceFileChanged(sceneFile);
	dispatchCommand(COMMAND_LOAD_SCENE);
}

void RenderViewport::setWidth(const QString &width) {
	window.setWidth(width.toInt());
	fixImageSize();
}
void RenderViewport::setHeight(const QString &height) {
	window.setHeight(height.toInt());
	fixImageSize();
}

void RenderViewport::enableButtons(bool enableStart, bool enableEnd, bool enableRestart) {
	ui.startButton->setEnabled(enableStart);
	ui.stopButton->setEnabled(enableEnd);
	ui.restartButton->setEnabled(enableRestart);
}
void RenderViewport::refreshIterationInfo(int iteration, double elapsed) {
	ui.iterationValue->setText(QString::number(iteration));
	ui.iterationTimeValue->setText((iteration > 0.0) ? QString::number(elapsed / ((double)iteration), 10, 4) : QString(""));
	long long elapsedSeconds = ((long long)elapsed);
	long long elapsedMinutes = (elapsedSeconds / 60); elapsedSeconds -= (60 * elapsedMinutes);
	long long elapsedHours = (elapsedMinutes / 60); elapsedMinutes -= (60 * elapsedHours);
	ui.elapsedTimeValue->setText(QString::number(elapsedHours) + ":" + QString::number(elapsedMinutes) + ":" + QString::number(elapsedSeconds));
}

void RenderViewport::startRender() {
	enableButtons(false, false, false);
	dispatchCommand(COMMAND_START_RENDER);
}
void RenderViewport::stopRender() {
	enableButtons(false, false, false);
	dispatchCommand(COMMAND_STOP_RENDER);
}
void RenderViewport::restartRender() {
	enableButtons(false, false, false);
	dispatchCommand(COMMAND_RESTART_RENDER);
}

void RenderViewport::setStatusText(const QString &text) {
	ui.image->setText(text);
}

void RenderViewport::resizeEvent(QResizeEvent *event) {
	QWidget::resizeEvent(event);
	fixImageSize();
}


void RenderViewport::emitUpdateImage(const QPixmap &map) {
	if (pixmapLock.try_lock()) {
		pixmap = map;
		pixmapLock.unlock();
		emit updateImage();
	}
}
void RenderViewport::updateImagePixmap(void *viewport, const QPixmap &map) {
	((RenderViewport*)viewport)->emitUpdateImage(map);
}


void RenderViewport::imageUpdated() {
	pixmapLock.lock();
	ui.image->setPixmap(pixmap);
	pixmapLock.unlock();
}


void RenderViewport::loadScene() {
	emit issueEnableButtons(false, false, false);
	//instance.interruptRender();
	//instance.stop();
	instance.~RenderInstance();
	emit issueSetStatusText(QString("Loading...\n[") + sceneFile.c_str() + "]");
	bool built = context.buildFromFile(sceneFile, &std::cout);
	new (&instance) DumbRenderContext::RenderInstance(&context, &window);
	instance.onIterationComplete(iterationEndedCallback, this);
	if (built) {
		instance.start();
		emit issueEnableButtons(false, true, true);
		emit issueSetStatusText("");
		ui.cpuThreadCount->setCurrentIndex(instance.cpuThreads());
	}
	else emit issueSetStatusText(QString("File is invalid...\n[") + sceneFile.c_str() + "]");
}
void RenderViewport::continueRender() {
	instance.start();
	emit issueEnableButtons(false, true, true);
}
void RenderViewport::interruptRender() {
	instance.stop();
	emit issueEnableButtons(true, false, true);
}
void RenderViewport::resetRender() {
	//instance.interruptRender();
	//instance.stop();
	instance.~RenderInstance();
	new (&instance) DumbRenderContext::RenderInstance(&context, &window);
	instance.onIterationComplete(iterationEndedCallback, this);
	instance.start();
	emit issueEnableButtons(false, true, true);
}


void RenderViewport::dispatchCommand(Command command) {
	std::lock_guard<std::mutex> guard(commandLock);
	commandQueue.emplace(command);
	commandCondition.notify_all();
}
void RenderViewport::controlThread(RenderViewport *viewport) {
	while (true) {
		Command command;
		while (true) {
			std::unique_lock<std::mutex> uniqueLock(viewport->commandLock);
			if (!viewport->commandQueue.empty()) {
				command = viewport->commandQueue.back();
				viewport->commandQueue.pop();
				break;
			}
			else viewport->commandCondition.wait(uniqueLock);
		}
		if (command == COMMAND_LOAD_SCENE) viewport->loadScene();
		else if (command == COMMAND_START_RENDER) viewport->continueRender();
		else if (command == COMMAND_STOP_RENDER) viewport->interruptRender();
		else if (command == COMMAND_RESTART_RENDER) viewport->resetRender();
		else if (command == COMMAND_UPDATE_TIMES) viewport->emit issueRefreshIterationInfo(viewport->instance.iteration(), viewport->instance.renderTime());
		else if (command == COMMAND_UPDATE_RESOURCES) viewport->updateResources();
		else if (command == COMMAND_QUIT) break;
	}
}
void RenderViewport::maintenanceThread(RenderViewport *viewport) {
	while ((viewport->flags & STATE_DESTRUCTOR_CALLED) == 0) {
		std::this_thread::sleep_for(std::chrono::milliseconds(512));
		viewport->dispatchCommand(COMMAND_UPDATE_TIMES);
	}
}
void RenderViewport::cpuThreadCountChanged(int) {
	dispatchCommand(COMMAND_UPDATE_RESOURCES);
}

void RenderViewport::updateResources() {
	// __TODO__: Complete with GPU stuff..
	instance.setCpuThreads(ui.cpuThreadCount->currentIndex());
}
