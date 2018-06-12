#include "TestingModule.cuh"
#include "DataStructures/DumbRenderContext/DumbRenderContext.cuh"
#include <iostream>
#include <thread>
#include <chrono>


int main(int argc, char *argv[]) {
	if (argc <= 1) return TestingModule::run();
	// Temporary, but anyway:
	else for (int i = 1; i < argc; i++)
		DumbRenderContext::testFile(argv[i]);
	return 0;
}
