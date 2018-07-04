#include "DumbRayQ.h"
#include <QtWidgets/QApplication>
#include "__source__/TestingModule.cuh"
#include "__source__/Namespaces/Dson/Dson.h"
#include "__source__/DataStructures/DumbRenderContext/DumbRenderContext.cuh"

int main(int argc, char *argv[])
{
	//TestingModule::run();
	QApplication a(argc, argv);
	DumbRayQ w;
	w.show();
	return a.exec();
}
