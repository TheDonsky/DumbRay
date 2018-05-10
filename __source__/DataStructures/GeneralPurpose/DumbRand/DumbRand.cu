#include"DumbRand.cuh"


std::mutex DumbRand::lock;


DumbRandHolder::DumbRandHolder() {
	cpuReference.count = 0;
	cpuReference.reference = NULL;
	if (cudaGetDeviceCount(&deviceCount) == cudaSuccess) {
		if (deviceCount > 0) gpuReferences = new DumbRandReference[deviceCount];
		else gpuReferences = NULL;
		if (gpuReferences == NULL) deviceCount = 0;
		for (int i = 0; i < deviceCount; i++) {
			DumbRandReference &ref = gpuReferences[i];
			ref.count = 0;
			ref.reference = NULL;
			ref.flags = 0;
		}
	}
	else {
		deviceCount = 0;
		gpuReferences = NULL;
	}
}
DumbRandHolder::~DumbRandHolder() {
	if (cpuReference.reference != NULL) delete[] cpuReference.reference;
	for (int i = 0; i < deviceCount; i++) {
		DumbRandReference &ref = gpuReferences[i];
		if (ref.reference != NULL) cudaFree(ref.reference);
		if ((ref.flags & DumbRandReference::STREAM_CREATED) != 0)
			cudaStreamDestroy(ref.stream);
	}
	if (gpuReferences != NULL) delete[] gpuReferences;
}

DumbRand *DumbRandHolder::getCPU(int count, bool lock) {
	if (lock) cpuReference.lock.lock();
	if (cpuReference.count < count) {
		DumbRand *newRefs = new DumbRand[count];
		if (newRefs == NULL) {
			if (lock) cpuReference.lock.unlock();
			return NULL;
		}
		for (int i = 0; i < cpuReference.count; i++)
			newRefs[i] = cpuReference.reference[i];
		for (int i = cpuReference.count; i < count; i++)
			newRefs[i].seed();
		delete[] cpuReference.reference;
		cpuReference.reference = newRefs;
		cpuReference.count = count;
	}
	if (lock) cpuReference.lock.unlock();
	return cpuReference.reference;
}
DumbRand *DumbRandHolder::getGPU(int count, int gpuId, bool lock) {
	if ((gpuId < 0) || (gpuId >= deviceCount)) return NULL;
	DumbRandReference &ref = gpuReferences[gpuId];
	if (lock) ref.lock.lock();
	if (ref.count < count) {
		if ((ref.flags & DumbRandReference::STREAM_CREATED) == 0)
			if (cudaStreamCreate(&ref.stream) == cudaSuccess)
				ref.flags |= DumbRandReference::STREAM_CREATED;
		if ((ref.flags & DumbRandReference::STREAM_CREATED) == 0) {
			if (lock) ref.lock.unlock();
			return NULL;
		}
		DumbRand *newRefsDevice;
		if (cudaMalloc((void**)&newRefsDevice, sizeof(DumbRand) * count) != cudaSuccess) {
			if (lock) ref.lock.unlock(); 
			return NULL;
		}
		bool success;
		if (ref.count > 0)
			success = (cudaMemcpyAsync((void*)newRefsDevice, (const void*)ref.reference,
				sizeof(DumbRand) * ref.count, cudaMemcpyDeviceToDevice, ref.stream) == cudaSuccess);
		else success = true;
		DumbRand *newRefsHost;
		if (success) {
			int hostRefCount = (count - ref.count);
			newRefsHost = new DumbRand[hostRefCount];
			if (newRefsHost != NULL) {
				for (int i = 0; i < hostRefCount; i++) newRefsHost[i].seed();
				success = (cudaMemcpyAsync((void*)(newRefsDevice + ref.count), (const void*)newRefsHost,
					sizeof(DumbRand) * hostRefCount, cudaMemcpyHostToDevice, ref.stream) == cudaSuccess);
			}
			else success = false;
		}
		else newRefsHost = NULL;
		if (cudaStreamSynchronize(ref.stream) != cudaSuccess) success = false;
		if (newRefsHost != NULL) delete[] newRefsHost;
		if (ref.reference != NULL) {
			success = (cudaFree(ref.reference) == cudaSuccess);
			ref.reference = NULL;
			ref.count = 0;
		}
		if (!success) {
			cudaFree(newRefsDevice);
			if (lock) ref.lock.unlock();
			return NULL;
		}
		else {
			ref.reference = newRefsDevice;
			ref.count = count; 
		}
	}
	if (lock) ref.lock.unlock();
	return ref.reference;
}
