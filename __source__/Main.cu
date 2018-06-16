#include "TestingModule.cuh"
#include "DataStructures/DumbRenderContext/DumbRenderContext.cuh"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>


int main(int argc, char *argv[]) {
	std::cout << std::fixed << std::setprecision(4);
	if (argc <= 1) return TestingModule::run();
	// Temporary, but anyway:
	else for (int i = 1; i < argc; i++)
		DumbRenderContext::testFile(argv[i]);
	return 0;
}
