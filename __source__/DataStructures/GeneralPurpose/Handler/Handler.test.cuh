#pragma once
#include "Handler.cuh"
#include "../Stacktor/Stacktor.cuh"
#include "../../../Namespaces/Tests/Tests.h"



namespace HandlerTest {
	namespace Private {
		__device__ __host__ inline void createAndDoShitWithHandler() {
			Handler<int> h;
			h.createHandle(1);
			if (h == 1) printf("Handler value correct...\n");
			else printf("Handler value incorrect!\n");
#ifndef __CUDA_ARCH__
			h.uploadHostHandleToDevice();
			h.destroyHandles();
#else
			h.destroyHandle();
#endif // __CUDA_ARCH__
#ifndef __CUDA_ARCH__
			Handler<Stacktor<int> > h2;
			system("PAUSE");
			h2.createHandle();
			h2.object().flush(256000000);
			system("PAUSE");
			h2.uploadHostHandleToDevice();
			system("PAUSE");
			h2.destroyHandles();
#endif // __CUDA_ARCH__
		}

		__global__ static void kernel() {
			createAndDoShitWithHandler();
		}
		inline static void test() {
			while (true) {
				std::cout << "Enter anthing to run Handler test: ";
				std::string s;
				std::getline(std::cin, s);
				if (s.length() <= 0) break;
				createAndDoShitWithHandler();
				kernel<<<1, 1>>>();
				cudaDeviceSynchronize();
			}
		}
	}

	inline static void test() {
		Tests::runTest(Private::test, "Testing Handler");
	}
}
