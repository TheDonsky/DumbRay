#include"Handler.cuh"




template<typename Type>
// Creates empty handler (both handles will be NULL)
__device__ __host__ inline Handler<Type>::Handler() {
	setNULL();
}
template<typename Type>
// Sets both references to NULL
__device__ __host__ inline void Handler<Type>::setNULL() {
	hostHandle = NULL;
	deviceHandle = NULL;
}
template<typename Type>
// Creates a handler with given pointers
__device__ __host__ inline Handler<Type>::Handler(Type *host, Type *dev) {
	hostHandle = host;
	deviceHandle = dev;
}
template<typename Type>
// Creates a handler, pointing to the given address (__host__ version will assign hostHandle, while __device__ version will set deviceHandle)
__device__ __host__ inline Handler<Type>::Handler(Type *address) {
#ifndef __CUDA_ARCH__
	hostHandle = address;
	deviceHandle = NULL;
#else
	hostHandle = NULL;
	deviceHandle = address;
#endif
}
template<typename Type>
// Assigns address to the handler (__host__ version will assign hostHandle, while __device__ version will set deviceHandle)
__device__ __host__ inline Handler<Type>& Handler<Type>::operator=(Type *address) {
#ifndef __CUDA_ARCH__
	hostHandle = address;
#else
	deviceHandle = address;
#endif
	return (*this);
}
template<typename Type>
// Cast to Type&
__device__ __host__ inline Handler<Type>::operator Type&() {
#ifndef __CUDA_ARCH__
	return (*hostHandle);
#else
	return (*deviceHandle);
#endif
}
template<typename Type>
// Cast to Type& (const)
__device__ __host__ inline Handler<Type>::operator const Type&()const {
#ifndef __CUDA_ARCH__
	return (*hostHandle);
#else
	return (*deviceHandle);
#endif
}
template<typename Type>
// Cast to Type*
__device__ __host__ inline Handler<Type>::operator Type*() {
#ifndef __CUDA_ARCH__
	return hostHandle;
#else
	return deviceHandle;
#endif
}
template<typename Type>
// Cast to Type* (const)
__device__ __host__ inline Handler<Type>::operator const Type*()const {
#ifndef __CUDA_ARCH__
	return hostHandle;
#else
	return deviceHandle;
#endif
}
template<typename Type>
// Cast to Type&
__device__ __host__ inline Type& Handler<Type>::object() {
	return ((Type&)(*this));
}
template<typename Type>
// Cast to Type& (const)
__device__ __host__ inline const Type& Handler<Type>::object()const {
	return ((const Type&)(*this));
}
template<typename Type>
// Cast to Type*
__device__ __host__ inline Type* Handler<Type>::pointer() {
	return ((Type*)(*this));
}
template<typename Type>
// Cast to Type* (const)
__device__ __host__ inline const Type* Handler<Type>::pointer()const {
	return ((const Type*)(*this));
}


template<typename Type>
template<typename... Args>
/*	Creates the new object with given arguments
Notes:
0. Returns true if and only if there's no previous reference and also, new instance was created successfuly;
1. Sets deviceHandle on __device__ and hostHandle on host.
*/
__device__ __host__ inline bool Handler<Type>::createHandle(const Args&... args) {
	if (pointer() != NULL) return false;
	Type *newAddr = new Type(args...);
	(*this) = newAddr;
	return (newAddr != NULL);
}
template<typename Type>
// Uploads the hostHandle to deviceHandle (returns true upon success)
__host__ inline bool Handler<Type>::uploadHostHandleToDevice(bool refreshExisting) {
	if (deviceHandle != NULL && (!refreshExisting)) return false;
	if (hostHandle == NULL) return false;

	cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
	Type *newDevHandle; 
	if (deviceHandle == NULL) {
		if (cudaMalloc(&newDevHandle, sizeof(Type)) != cudaSuccess) { cudaStreamDestroy(stream); return false; }
	}
	else {
		newDevHandle = deviceHandle;
		if (!TypeTools<Type>::disposeDevArray(newDevHandle, 1)) { cudaStreamDestroy(stream); return false; }
	}

	char garbage[sizeof(Type)];
	Type* hosClone = ((Type*)garbage);
	bool success = TypeTools<Type>::prepareForCpyLoad(hostHandle, hosClone, newDevHandle, 1);
	if (success) {
		success = (cudaMemcpyAsync(newDevHandle, hosClone, sizeof(Type), cudaMemcpyHostToDevice, stream) == cudaSuccess);
		if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;
		if (!success) TypeTools<Type>::undoCpyLoadPreparations(hostHandle, hosClone, newDevHandle, 1);
	}

	if (cudaStreamDestroy(stream) != cudaSuccess) {
		if (success) TypeTools<Type>::disposeDevArray(newDevHandle, 1);
		success = false;
	}
	if (success) deviceHandle = newDevHandle;
	else {
		cudaFree(newDevHandle);
		deviceHandle = NULL;
	}
	return success;
}
template<typename Type>
// Destroys handle (hostHandle on __host__ and deviceHandle on __device__)
__device__ __host__ inline bool Handler<Type>::destroyHandle() {
	if (pointer() == NULL) return true;
	delete pointer();
	(*this) = NULL;
	return true;
}
template<typename Type>
// Destroys hostHandle
__host__ inline bool Handler<Type>::destroyHostHandle() {
	return destroyHandle();
}
template<typename Type>
// Destroys deviceHandle
__host__ inline bool Handler<Type>::destroyDeviceHandle() {
	if (deviceHandle == NULL) return true;
	if (!TypeTools<Type>::disposeDevArray(deviceHandle, 1)) return false;
	if (cudaFree(deviceHandle) != cudaSuccess) return false;
	deviceHandle = NULL;
	return true;
}
template<typename Type>
// Destroys both handles
__host__ inline bool Handler<Type>::destroyHandles() {
	bool success = destroyHostHandle();
	if (!destroyDeviceHandle()) success = false;
	return success;
}




