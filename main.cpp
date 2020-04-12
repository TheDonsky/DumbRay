#include "DumbRayQ.h"
#ifndef NDEBUG
#include <thread>
#endif
#include <QtWidgets/qapplication.h>
#include "__source__/TestingModule.cuh"

int main(int argc, char *argv[])
{
#ifndef NDEBUG
	std::thread testThread(TestingModule::run);
#endif
	QApplication a(argc, argv);
	DumbRayQ w(Q_NULLPTR, (argc >= 2) ? argv[1] : NULL);
	w.show();
	int rv = a.exec();
#ifndef NDEBUG
	testThread.join();
#endif
	return rv;
}
