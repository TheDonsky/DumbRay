#include "DumbRayQ.h"
#ifndef NDEBUG
#include <thread>
#endif
#include <QtWidgets/qapplication.h>
#include "__source__/TestingModule.cuh"
#include <windows.h>

int main(int argc, char *argv[])
{
	bool hasTests = false;
	for (size_t i = 0u; i < argc; i++)
		if (std::string(argv[i]) == "--test" || std::string(argv[i]) == "-t")
			hasTests = true;
	if (hasTests) {
		AllocConsole();
		FILE* fp;
		freopen_s(&fp, "CONOUT$", "w", stdout);
		freopen_s(&fp, "CONIN$", "r", stdin);
		TestingModule::run();
		return 0;
	}
	else {
		QApplication a(argc, argv);
		DumbRayQ w(Q_NULLPTR, (argc >= 2) ? argv[1] : NULL);
		w.show();
		int rv = a.exec();
		return rv;
	}
}
