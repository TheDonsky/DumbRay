#include "DumbRayQ.h"

DumbRayQ::DumbRayQ(QWidget *parent, const char *filename) : QMainWindow(parent)
{
	ui.setupUi(this);
	setWindowTitle("DumbRay");
	RenderViewport *viewport = new RenderViewport(this, filename);
	connect(viewport, SIGNAL(sourceFileChanged(const std::string &)), this, SLOT(viewportSouceFilechanged(const std::string &)));
	if (filename != NULL) viewportSouceFilechanged(filename);
	ui.centralWidget->layout()->addWidget(viewport);
	ui.mainToolBar->hide();
	ui.statusBar->hide();
	ui.menuBar->hide();
}

void DumbRayQ::viewportSouceFilechanged(const std::string &filename) {
	setWindowTitle("DumbRay [" + QString(filename.c_str()) + "]");
}
