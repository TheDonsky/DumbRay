#include "DumbRayQ.h"
#include <QtWidgets/QApplication>
#include "__source__/TestingModule.cuh"
//#include "__source__/DataStructures/DumbRenderContext/DumbRenderContextConnector.cuh"

int main(int argc, char *argv[])
{
	TestingModule::run();
	//DumbRenderContext context;
	QApplication a(argc, argv);
	DumbRayQ w;
	w.show();
	return a.exec();
}
