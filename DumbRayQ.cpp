#include "DumbRayQ.h"

DumbRayQ::DumbRayQ(QWidget *parent, const char *filename)
	: QMainWindow(parent), viewport(Q_NULLPTR, filename)
{
	ui.setupUi(this);
	ui.centralWidget->layout()->addWidget(&viewport);
}
