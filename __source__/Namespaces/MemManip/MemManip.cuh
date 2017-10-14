#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<vector>
#include<thread>

namespace MemManip{
	template<typename Type>
	// Creates a clone of the given array
	__device__ __host__ inline static Type* arrayClone(const Type *arr, const int size);
	
	template<typename Type>
	// Creates a clone of the given std::vector
	inline static Type* vectorClone(const std::vector<Type> &vec);
	
	template<typename Type>
	// Swaps the values of the given items
	__device__ __host__ inline static void swap(Type &a, Type &b);
	
	template<typename Type>
	// Swaps the content of the given items bit by bit
	__device__ __host__ inline static void bitwiseSwap(Type &a, Type &b);

	template<typename Type>
	// Transfers the given data from the source to the destination
	// Note: If dataTransferFunction is NULL/not specified, operator= will be used.
	__device__ __host__ inline static void transferData(Type *destination, Type *source, const int size, void(*dataTransferFunction)(Type &dst, Type &src) = NULL);

	template<typename Type>
	// Transfers the given data from the source to the destination using multiple threads
	// Note: If numThreads is equal to 0, 1 thread will be used.
	inline static void transferDataMultiThread(Type *destination, Type *source, const int size, int numThreads, void(*dataTransferFunction)(Type &dst, Type &src) = NULL);

	template<typename Type>
	// Doubles the capacity of the given array
	// Notes:	nonDeletableAddr is an optional parameter, to assert that nothing's being deleted, if it's not permitted to;
	//			Multithreaded data transfer available only for the host version.
	__device__ __host__ inline static bool doubleCapacity(Type *&arr, const int elemsUsed, int &currentCapacity, void(*dataTransferFunction)(Type &dst, Type &src) = NULL, Type *nonDeletableAddr = NULL, int numTransferThreads = 0);
}


#include"MemManip.impl.cuh"
