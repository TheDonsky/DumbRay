#pragma once

#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<math.h>
#include<mutex>
#include<new>
#include"DataStructures/GeneralPurpose/Stacktor/Stacktor.cuh"


class Cutex{
public:
	__device__ __host__ inline Cutex();
	__device__ __host__ inline void initRaw();
	__device__ __host__ inline ~Cutex();
	__device__ __host__ inline void disposeRaw();
	__device__ __host__ inline void init(int startState = 1);
	__device__ __host__ inline void lock();
	__device__ __host__ inline void unlock();
	template<typename ReturnType = int, typename Function, typename... Args>
	__device__ __host__ inline ReturnType atomicCall(Function&& func, Args&... args);


private:
	volatile int state;
	volatile char mutexBytes[sizeof(std::mutex)];
};




#include"Cutex.impl.cuh"
