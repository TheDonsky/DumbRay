#include "TestingModule.cuh"
#include <iostream>
#include <thread>
#include <chrono>


int main(int argc, char *argv[]) {
	if (argc <= 1) return TestingModule::run();
	else {
		std::cout << "Parameters: " << std::endl;
		for (int i = 0; i < argc; i++)
			std::cout << i << ". \"" << argv[i] << "\"" << std::endl;
		std::cout << std::endl << "NO IDEA, WHAT TO DO WITH THESE..." << std::endl;
		while (true) std::this_thread::sleep_for(std::chrono::seconds(1));
	}
	return 0;
}
