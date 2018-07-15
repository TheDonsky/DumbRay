#include "DumbRayQ.h"

DumbRayQ::DumbRayQ(QWidget *parent, const char *filename) : QMainWindow(parent)
{
	ui.setupUi(this);
	setWindowTitle("DumbRay");
	RenderViewport *viewport = new RenderViewport(this, filename);
	ui.centralWidget->layout()->addWidget(viewport);
	ui.mainToolBar->hide();
	ui.statusBar->hide();
	ui.menuBar->hide();
	connect(viewport, SIGNAL(sourceFileChanged(const std::string &)), this, SLOT(souceFilechanged(const std::string &)));
}

void DumbRayQ::souceFilechanged(const std::string &filename) {
	setWindowTitle("DumbRay [" + QString(filename.c_str()) + "]");
}
