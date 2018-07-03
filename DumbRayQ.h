#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_DumbRayQ.h"

class DumbRayQ : public QMainWindow
{
	Q_OBJECT

public:
	DumbRayQ(QWidget *parent = Q_NULLPTR);

private:
	Ui::DumbRayQClass ui;
};
