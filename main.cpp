#include "DumbRayQ.h"
#include <QtWidgets/qapplication.h>
#include "__source__/TestingModule.cuh"

int main(int argc, char *argv[])
{
	//TestingModule::run();
	QApplication a(argc, argv);
	DumbRayQ w(Q_NULLPTR, (argc >= 2) ? argv[1] : NULL);
	w.show();
	return a.exec();
}
