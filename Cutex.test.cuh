#pragma once

#include"Cutex.cuh"
#include"Tests.h"
#include<thread>



namespace CutexTest{
	namespace Private{
		
		__device__ __host__ inline static int increase(volatile int &value){
			value++;
			return value;
		}

		inline static void hostThread(volatile int *value, Cutex *mutex, int iterations){
			for (int i = 0; i < iterations; i++)
				mutex->atomicCall(increase, *value);
		}

		inline static void testHost(){
			volatile int value = 0;
			int iterationsPerThread = 1024;
			const int numThreads = 32;
			std::thread threads[32];
			Cutex mutex;
			for (int i = 0; i < numThreads; i++)
				threads[i] = std::thread(hostThread, &value, &mutex, iterationsPerThread);
			for (int i = 0; i < numThreads; i++)
				threads[i].join();
			if (value == iterationsPerThread * numThreads)
				std::cout << "PASSED" << std::endl;
			else std::cout << "FAILED: value is " << value << " instead of " << iterationsPerThread * numThreads << std::endl;
		}


		__global__ void increment(int *value, Cutex *c){
			c->atomicCall(increase, *value);
		}

		__global__ void init(Cutex *c){
			c->initRaw();
		}

		inline static void testDevice(){
			int *value; if (cudaMalloc(&value, sizeof(int)) != cudaSuccess){
				std::cout << "ERROR ALLOCATING VALUE" << std::endl;
				return;
			}
			Cutex *mutex; if (cudaMalloc(&mutex, sizeof(Cutex)) != cudaSuccess){
				cudaFree(value);
				std::cout << "ERROR ALLOCATING CUTEX" << std::endl;
				return;
			}
			init<<<1, 1>>>(mutex);
			if ((cudaDeviceSynchronize() == cudaSuccess)){
				const int threads = 16;
				const int blocks = 16;
				increment<<<threads, blocks>>>(value, mutex);
				if (cudaDeviceSynchronize() == cudaSuccess){
					int loadedValue;
					if (cudaMemcpy(&loadedValue, value, sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess){
						if (loadedValue == threads * blocks)
							std::cout << "PASSED" << std::endl;
						else std::cout << "FAILED: value is " << loadedValue << " instead of " << threads * blocks << std::endl;
					}
					else std::cout << "ERROR LOADING THE RESULT" << std::endl;
				}
				else std::cout << "ERROR SINCHRONISING" << std::endl;
			}
			else std::cout << "ERROR INITIALISING CUTEX" << std::endl;
			cudaFree(value);
			cudaFree(mutex);
		}

		inline static void test(){
			Tests::call("Testing on host", testHost);
			Tests::call("Testing on device", testDevice);
		}
	}
	
	inline static void test(){
		Tests::runTest(Private::test, "Running Cutex test");
	}
}

