#include "ManagedHandler.cuh"




template <typename Type>
/*
Constructor.
*/
inline ManagedHandler<Type>::ManagedHandler(const Type &s) {
	data = (&s);
	int deviceCount;
	if (cudaGetDeviceCount(&deviceCount) == cudaSuccess)
		for (int i = 0; i < deviceCount; i++)
			deviceData.push(NULL);
}

template <typename Type>
/*
Destructor.
*/
inline ManagedHandler<Type>::~ManagedHandler() {
	cleanEveryGPU();
}

template <typename Type>
/*
Sets the GPU context.
*/
inline bool ManagedHandler<Type>::selectGPU(int index)const {
	if (index < 0 || index >= deviceData.size()) return false;
	else return (cudaSetDevice(index) == cudaSuccess);
}

template <typename Type>
/*
Uploads/updates the item to the given GPU and selects it's context.
*/
inline bool ManagedHandler<Type>::uploadToGPU(int index, bool overrideExisting) {
	std::lock_guard<std::mutex> guard(lock);
	//if (selectGPU(index)) {
		if (deviceData[index] != NULL) {
			if (overrideExisting) {
				if (!TypeTools<Type>::disposeDevArray(deviceData[index], 1)) return false;
			}
			else return true;
		}
		else if (cudaMalloc(deviceData + index, sizeof(Type)) != cudaSuccess) return false;
		cudaStream_t stream; 
		if (cudaStreamCreate(&stream) != cudaSuccess) {
			cudaFree(deviceData[index]);
			deviceData[index] = NULL;
			return false;
		}
		char garbage[sizeof(Type)];
		Type *hosClone = ((Type*)garbage);
		bool success = TypeTools<Type>::prepareForCpyLoad(data, hosClone, deviceData[index], 1);
		if (success) {
			success = (cudaMemcpyAsync(deviceData[index], hosClone, sizeof(Type), cudaMemcpyHostToDevice, stream) == cudaSuccess);
			if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;
			if (!success) TypeTools<Type>::undoCpyLoadPreparations(data, hosClone, deviceData[index], 1);
		}
		if (cudaStreamDestroy(stream) != cudaSuccess) {
			if (success) TypeTools<Type>::disposeDevArray(deviceData[index], 1);
			success = false;
		}
		if (!success) {
			cudaFree(deviceData[index]);
			deviceData[index] = NULL;
		}
		return success;
	//}
	//else return false;
}

template <typename Type>
/*
Deallocates the GPU instance (leaves context set).
*/
inline bool ManagedHandler<Type>::cleanGPU(int index) {
	std::lock_guard<std::mutex> guard(lock);
	//if (selectGPU(index)) {
		return cleanDeviceInstanceNoLock(index);
	//}
	//else return false;
}

template <typename Type>
/*
Uploads/updates the item on every available GPU (which context will stay selected, is undefined).
*/
inline void ManagedHandler<Type>::uploadToEveryGPU(bool overrideExisting) {
	for (int i = 0; i < deviceData.size(); i++) uploadToGPU(i, overrideExisting);
}

template <typename Type>
/*
Deallocates every GPU instance (which context will stay selected, is undefined).
*/
inline void ManagedHandler<Type>::cleanEveryGPU() {
	if (deviceData.size() <= 0) return;
	std::lock_guard<std::mutex> guard(lock);
	std::thread *threads = new std::thread[deviceData.size()];
	for (int i = 0; i < deviceData.size(); i++) threads[i] = std::thread(cleanDeviceInstanceThread, this, i);
	for (int i = 0; i < deviceData.size(); i++) threads[i].join();
	delete[] threads;
	//for (int i = 0; i < deviceData.size(); i++) cleanGPU(i);
}

template <typename Type>
inline void ManagedHandler<Type>::cleanDeviceInstanceThread(ManagedHandler *self, int deviceId) {
	if (self->selectGPU(deviceId)) self->cleanDeviceInstanceNoLock(deviceId);
}
template <typename Type>
inline bool ManagedHandler<Type>::cleanDeviceInstanceNoLock(int index) {
	if (deviceData[index] != NULL) {
		if (!TypeTools<Type>::disposeDevArray(deviceData[index], 1)) return false;
		else if (cudaFree(deviceData[index]) != cudaSuccess) return false;
		else {
			deviceData[index] = NULL;
			return true;
		}
	}
	else return true;
}

template <typename Type>
/*
Returns detected GPU count.
*/
inline int ManagedHandler<Type>::gpuCount()const {
	return (deviceData.size());
}

template <typename Type>
/*
Returns CPU handle.
*/
inline const Type* ManagedHandler<Type>::getHandleCPU()const {
	return data;
}

template <typename Type>
/*
Returns GPU handle (no context selection here...).
*/
inline Type* ManagedHandler<Type>::getHandleGPU(int index) {
	return deviceData[index];
}

template <typename Type>
/*
Returns GPU handle (no context selection here...).
*/
inline const Type* ManagedHandler<Type>::getHandleGPU(int index)const {
	return deviceData[index];
}



