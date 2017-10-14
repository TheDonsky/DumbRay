#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"../TypeTools/TypeTools.cuh"




template<typename Type>
struct Handler {
	Type *hostHandle;
	Type *deviceHandle;


	// Creates empty handler (both handles will be NULL)
	__device__ __host__ inline Handler();
	// Sets both references to NULL
	__device__ __host__ inline void setNULL();
	// Creates a handler with given pointers
	__device__ __host__ inline Handler(Type *host, Type *dev);
	// Creates a handler, pointing to the given address (__host__ version will assign hostHandle, while __device__ version will set deviceHandle)
	__device__ __host__ inline Handler(Type *address);
	// Assigns address to the handler (__host__ version will assign hostHandle, while __device__ version will set deviceHandle)
	__device__ __host__ inline Handler& operator=(Type *address);
	// Cast to Type&
	__device__ __host__ inline operator Type&();
	// Cast to Type& (const)
	__device__ __host__ inline operator const Type&()const;
	// Cast to Type*
	__device__ __host__ inline operator Type*();
	// Cast to Type* (const)
	__device__ __host__ inline operator const Type*()const;
	// Cast to Type&
	__device__ __host__ inline Type& object();
	// Cast to Type& (const)
	__device__ __host__ inline const Type& object()const;
	// Cast to Type*
	__device__ __host__ inline Type* pointer();
	// Cast to Type* (const)
	__device__ __host__ inline const Type* pointer()const;


	template<typename... Args>
	/*	Creates the new object with given arguments
		Notes:
			0. Returns true if and only if there's no previous reference and also, new instance was created successfuly;
			1. Sets deviceHandle on __device__ and hostHandle on host.
	*/
	__device__ __host__ inline bool createHandle(const Args&... args);
	// Uploads the hostHandle to deviceHandle (returns true upon success)
	__host__ inline bool uploadHostHandleToDevice(bool refreshExisting = false);
	// Destroys handle (hostHandle on __host__ and deviceHandle on __device__)
	__device__ __host__ inline bool destroyHandle();
	// Destroys hostHandle
	__host__ inline bool destroyHostHandle();
	// Destroys deviceHandle
	__host__ inline bool destroyDeviceHandle();
	// Destroys both handles
	__host__ inline bool destroyHandles();
};







#include"Handler.impl.cuh"

