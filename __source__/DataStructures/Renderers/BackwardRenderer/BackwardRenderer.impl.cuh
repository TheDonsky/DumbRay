#include "BackwardRenderer.cuh"




/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
inline BackwardRenderer<HitType>::BackwardRenderer(const ThreadConfiguration &threads, SceneHandler<HitType> &scene) : Renderer(threads){
	sceneData = (&scene);
	selectedCamera = 0;
}
template<typename HitType>
inline BackwardRenderer<HitType>::~BackwardRenderer() {
	killRenderThreads();
}


template<typename HitType>
inline bool BackwardRenderer<HitType>::selectCamera(int index) {
	if (index >= 0 && index < sceneData->getHandleCPU()->cameras.size()) {
		selectedCamera = index;
		resetIterations();
		return true;
	}
	else return false;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
inline bool BackwardRenderer<HitType>::setupSharedData(const Info &info, void *&sharedData) {
	if (info.isGPU()) return sceneData->uploadToGPU(info.device, false);
	else return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::setupData(const Info &info, void *&data) {
	if (info.isGPU()) {
		if (sceneData->getHandleGPU(info.device) == NULL) return false;
		if ((!info.manageSharedData) && (!sceneData->selectGPU(info.device))) return false;
		DeviceThreadData *deviceThreadData = new DeviceThreadData();
		if (deviceThreadData != NULL) {
			if (deviceThreadData->pixels != NULL && deviceThreadData->renderEnded != NULL) {
				data = ((void*)deviceThreadData);
				return true;
			}
			else {
				delete deviceThreadData;
				return false;
			}
		}
		else return false;
	}
	else {
		if (sceneData->getHandleCPU() == NULL) return false;
		HostThreadData *hostThreadData = new HostThreadData();
		if (hostThreadData != NULL) {
			if (hostThreadData->pixels != NULL) {
				data = ((void*)hostThreadData);
				return true;
			}
			else {
				delete hostThreadData;
				return false;
			}
		}
		else return false;
	}
	return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::prepareIteration() {
	segmentBank.reset(1000);
	// __TODO__
	return true;
}
template<typename HitType>
inline void BackwardRenderer<HitType>::iterateCPU(const Info &info) {
	HostThreadData &threadData = ((HostThreadData*)info.data);
	int start, end;
	while (segmentBank.get(1, start, end)) {
		// __TODO__
	}
}
template<typename HitType>
inline void BackwardRenderer<HitType>::iterateGPU(const Info &info) {
	// __TODO__
	DeviceThreadData &threadData = ((DeviceThreadData*)info.data);
	int start, end;
	while (segmentBank.get(threadData.blockCount, start, end)) {
		// __TODO__
	}
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::completeIteration() {
	// __TODO__
	return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::clearData(const Info &info, void *&data) {
	if (data != NULL) {
		if (info.isGPU()) {
			DeviceThreadData *deviceThreadData = ((DeviceThreadData*)data);
			delete deviceThreadData;
			data = NULL;
		}
		else {
			HostThreadData *hostThreadData = ((HostThreadData*)data);
			delete hostThreadData;
			data = NULL;
		}
	}
	return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::clearSharedData(const Info &info, void *&sharedData) {
	// __TODO__
	return true;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
__dumb__ void BackwardRenderer<HitType>::PixelRenderProcess::init(const Scene<HitType> &scn, const Camera &cmr, int x, int y, int w, int h, BoxingType boxing, int maxBnc) {
	scene = &scn;
	pixelColor = Color(0, 0, 0, 0);
	bounces.clear();
	lightIllumination.clear();
	posX = x;
	posY = y;
	maxBounces = maxBnc;
	bounces.flush(1);
	BounceObject &bounce = bounces.top();
	bounce.hit.object = NULL;
	bounce.color = ColorRGB(0, 0, 0);
	float divisor;
	if (boxing == VERTICAL) divisor = y;
	else if (boxing == HORIZONTAL) divisor = x;
	else if (boxing == MINIMAL) divisor = min(x, y);
	else if (boxing == MAXIMAL) divisor = max(x, y);
	else divisor = 1;
	float scale = (1.0f / divisor);
	cmr.getPhoton(Vector2((((float)x) - (((float)w) / 2.0f)) * scale, ((((float)h) / 2.0f) - ((float)y)) * scale), bounce.bounce);
	// __TODO__ ?
}

template<typename HitType>
__dumb__ bool BackwardRenderer<HitType>::PixelRenderProcess::render(int &raycastBudget, int &lightCallBudget) {
	// __TODO__
	return true;
}





namespace __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__ {
	namespace DeviceDataManagementKernels {
		__dumb__ int blockSize() { return 128; }
		__device__ inline int index() { return ((blockIdx.x * blockDim.x) + threadIdx.x); }
		template<typename HitType>
		__global__ inline void callPixelRenderProcessConstructors(BackwardRenderer<HitType>::PixelRenderProcess *pixels, int count) {
			int id = index(); if (id < count) new(pixels + id) typename BackwardRenderer<HitType>::PixelRenderProcess();
		}
		template<typename HitType>
		__global__ inline void callPixelRenderProcessDestructors(BackwardRenderer<HitType>::PixelRenderProcess *pixels, int count) {
			int id = index(); if (id < count) (pixels + id)->~PixelRenderProcess();
		}
	}
}
template<typename HitType>
inline BackwardRenderer<HitType>::DeviceThreadData::DeviceThreadData(int blkWidth, int blkHeight) {
	pixels = NULL;
	renderEnded = NULL;
	if (cudaStreamCreate(&stream) != cudaSuccess) return;
	blockCount = 1024; // This one has to be changed to something else...
	blockWidth = blkWidth;
	blockHeight = blkHeight;
	if (cudaMalloc(&pixels, sizeof(PixelRenderProcess) * pixelCount()) != cudaSuccess) {
		cudaStreamDestroy(stream);
		pixels = NULL;
		return;
	}
	if (cudaHostAlloc(&renderEnded, sizeof(bool), cudaHostAllocMapped) != cudaSuccess) {
		cudaStreamDestroy(stream);
		cudaFree(pixels);
		pixels = NULL;
		renderEnded = NULL;
		return;
	}
	int threads = __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
		::DeviceDataManagementKernels::blockSize();
	int blocks = ((pixelCount + threads - 1) / threads);
	__DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
		::DeviceDataManagementKernels
		::callPixelRenderProcessConstructors<HitType><<<blocks, threads, 0, stream>>>(pixels, pixelCount);
	if (cudaStreamSynchronize(stream) != cudaSuccess) {
		cudaStreamDestroy(stream);
		cudaFreeHost(renderEnded);
		cudaFree(pixels);
		pixels = NULL;
		renderEnded = NULL;
		return;
	}
	raycastBudget = 1;
	lightCallBudget = 256;
}
template<typename HitType>
inline BackwardRenderer<HitType>::DeviceThreadData::~DeviceThreadData() {
	bool streamExists = false;
	if (pixels != NULL) {
		int threads = __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
			::DeviceDataManagementKernels::blockSize();
		int blocks = ((pixelCount + threads - 1) / threads);
		__DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
			::DeviceDataManagementKernels
			::callPixelRenderProcessDestructors<HitType><<<blocks, threads, 0, stream>>>(pixels, pixelCount);
		cudaStreamSynchronize(stream);
		cudaFree(pixels);
		pixels = NULL;
		streamExists = true;
	}
	if (renderEnded != NULL) {
		cudaFreeHost(renderEnded);
		renderEnded = NULL;
		streamExists = true;
	}
	if (streamExists) cudaStreamDestroy(stream);
}

template<typename HitType>
inline int BackwardRenderer<HitType>::DeviceThreadData::pixelCount()const {
	return (blockCount * blockWidth * blockHeight);
}





template<typename HitType>
inline BackwardRenderer<HitType>::HostThreadData::HostThreadData(int blkWidth, int blkHeight) {
	blockWidth = blkWidth;
	blockHeight = blkHeight;
	pixels = new PixelRenderProcess[pixelCount()];
}
template<typename HitType>
inline BackwardRenderer<HitType>::HostThreadData::~HostThreadData() {
	if (pixels != NULL) {
		delete[] pixels;
		pixels = NULL;
	}
}
template<typename HitType>
inline int BackwardRenderer<HitType>::HostThreadData::pixelCount()const {
	return (blockWidth * blockHeight);
}


template<typename HitType>
inline void BackwardRenderer<HitType>::SegmentBank::reset(int totalBlocks) {
	blockCount = totalBlocks;
	counter = 0;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::SegmentBank::get(int count, int &start, int &end) {
	lock.lock();
	bool canAllocate = (counter < blockCount);
	if (canAllocate) {
		start = counter;
		counter += count;
		end = min(blockCount, counter);
	}
	lock.unlock();
	return canAllocate;
}

