#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DumbRayQ.h"
#include "__qt_source__/RenderViewport.h"

class DumbRayQ : public QMainWindow
{
	Q_OBJECT

public:
	DumbRayQ(QWidget *parent = Q_NULLPTR, const char *filename = NULL);

private:
	Ui::DumbRayQClass ui;
	RenderViewport viewport;
};
