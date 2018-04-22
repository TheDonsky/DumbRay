#include "BackwardRenderer.cuh"




/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
inline BackwardRenderer<HitType>::Configuration::Configuration(
	SceneHandler<HitType> &s,
	int maxBounce,
	int samples,
	int blockWidth, int blockHeight, 
	BoxingType boxing,
	int camera, 
	int devOversaturation,
	int pixelPerDeviceThread,
	int cpuOversaturation) {
	scene = (&s);
	maxBounces = maxBounce;
	multisampling = samples;
	segmentWidth = blockWidth;
	segmentHeight = blockHeight;
	boxingType = boxing;
	cameraId = camera;
	deviceOverstauration = devOversaturation;
	devicePixelsPerThread = pixelPerDeviceThread;
	hostOversaturation = cpuOversaturation;
}

template<typename HitType>
inline BackwardRenderer<HitType>::BackwardRenderer(
	const Configuration &configuration, const ThreadConfiguration &threads) 
	: Renderer(threads), config(configuration) { 
	frameBuffer = NULL;
}
template<typename HitType>
inline BackwardRenderer<HitType>::~BackwardRenderer() { killRenderThreads(); }


template<typename HitType>
inline bool BackwardRenderer<HitType>::selectCamera(int index) {
	if (index >= 0 && index < config.scene->getHandleCPU()->cameras.size()) {
		config.cameraId = index;
		resetIterations();
		return true;
	}
	else return false;
}
template<typename HitType>
inline void BackwardRenderer<HitType>::setBoxingType(BoxingType boxing) {
	config.boxingType = boxing;
}
template<typename HitType>
inline void BackwardRenderer<HitType>::setFrameBuffer(FrameBufferManager &manager) {
	if (frameBuffer != (&manager)) {
		frameBuffer = (&manager);
		resetIterations();
	}
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
inline bool BackwardRenderer<HitType>::setupSharedData(const Info &info, void *&sharedData) {
	if (info.isGPU()) {
		if (!config.scene->uploadToGPU(info.device, false)) return false;
		//*
		size_t stackSize;
		if (cudaDeviceGetLimit(&stackSize, cudaLimitStackSize) != cudaSuccess) return false;
		const int neededStackSize = 8096; // (sizeof(PixelRenderProcess) + 256);
		if (stackSize < neededStackSize) 
			return (cudaDeviceSetLimit(cudaLimitStackSize, neededStackSize) == cudaSuccess);
		//*/
		else return true;
	}
	else return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::setupData(const Info &info, void *&data) {
	if (info.isGPU()) {
		if (config.scene->getHandleGPU(info.device) == NULL) return false;
		if ((!info.manageSharedData) && (!config.scene->selectGPU(info.device))) return false;
		DeviceThreadData *deviceThreadData = new DeviceThreadData(this);
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
		if (config.scene->getHandleCPU() == NULL) return false;
		else return true;
	}
}
namespace {
	namespace __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__ {
		namespace Pixels {
			__dumb__ int getBlockCount(int size, int blockSize) {
				register int value = ((size + blockSize - 1) / blockSize);
				return ((value > 0) ? value : 0);
			}
			__dumb__ int getMaxBounces(int desired) {
				if (desired < 0) return 0;
				else if (desired > BACKWARD_RENDERER_MAX_BOUNCES) return BACKWARD_RENDERER_MAX_BOUNCES;
				else return desired;
			}
			__dumb__ bool getPosition(
				int block, int pixel,
				int blockWidth, int blockHeight,
				int width, int height,
				int &x, int &y) {
				if (pixel < 0 || pixel >= (blockWidth * blockHeight)) return false;
				int wBlocks = getBlockCount(width, blockWidth);
				int hBlocks = getBlockCount(height, blockHeight);
				if (block < 0 || block >= (wBlocks * hBlocks)) return false;
				int blockPosY = (block / wBlocks);
				int blockPosX = (block - (wBlocks * blockPosY));
				int pixelY = (pixel / blockWidth);
				int pixelX = (pixel - (blockWidth * pixelY));
				register int posX = ((blockPosX * blockWidth) + pixelX);
				if (posX >= width) return false;
				register int posY = ((blockPosY * blockHeight) + pixelY);
				if (posY >= height) return false;
				x = posX;
				y = posY;
				return true;
			}
		}
	}
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::prepareIteration() {
	if (frameBuffer != NULL) {
		frameBuffer->lockEdit();
		int width, height;
		frameBuffer->cpuHandle()->getSize(&width, &height);
		int wBlocks = __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
			::Pixels::getBlockCount(width, config.segmentWidth);
		int hBlocks = __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
			::Pixels::getBlockCount(height, config.segmentHeight);
		segmentBank.reset(wBlocks * hBlocks);
		return true;
	}
	else return false;
}
template<typename HitType>
inline void BackwardRenderer<HitType>::iterateCPU(const Info &info) {
	PixelRenderProcess pixelRenderProcess;

	int maxBounce = __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
		::Pixels::getMaxBounces(config.maxBounces);
	pixelRenderProcess.settings = PixelRenderProcess::Settings(
		maxBounce, 
		((config.multisampling > 1) ? config.multisampling : 1), 
		config.boxingType,
		(1.0f / ((float)iteration())));
	pixelRenderProcess.context = PixelRenderProcess::Context(
		config.scene->getHandleCPU(), 
		config.scene->getHandleCPU()->cameras + config.cameraId, 
		frameBuffer->cpuHandle());

	int start, end;
	int blockSize = (config.segmentWidth * config.segmentHeight);
	while (segmentBank.get(max(config.hostOversaturation, 1), start, end))
		for (int block = start; block < end; block++)
			for (int i = 0; i < blockSize; i++) {
				pixelRenderProcess.domain = PixelRenderProcess::Domain(
					block, block + 1, config.segmentWidth, config.segmentHeight, i);
				pixelRenderProcess.resetState();
				while (true) {
					int raycastBudget = 1000000;
					if (pixelRenderProcess.render(raycastBudget)) break;
				}
			}
}
namespace {
	namespace __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__ {
		namespace RenderKernels {
			__device__ inline int getIndex() { return ((blockIdx.x * blockDim.x) + threadIdx.x); }
			template<typename HitType>
			__global__ static void render(
				BackwardRenderer<HitType>::PixelRenderProcess *processes,
				int raycastBudget, bool *renderEnded) {
				typename BackwardRenderer<HitType>::PixelRenderProcess process = processes[getIndex()];
				if (!process.render(raycastBudget)) {
					(*renderEnded) = false;
					processes[getIndex()] = process;
				}
			}
			template<typename HitType>
			__global__ static void init(
				BackwardRenderer<HitType>::PixelRenderProcess *processes,
				BackwardRenderer<HitType>::PixelRenderProcess::Settings settings,
				Scene<HitType> *scene, int cameraId, FrameBuffer *frameBuffer,
				int startBlock, int endBlock, int blocksPerThread, 
				int blockWidth, int blockHeight,
				Transform transform, int raycastBudget, bool *renderEnded) {
				
				typename BackwardRenderer<HitType>::PixelRenderProcess &process = processes[getIndex()];
				Camera *camera = &scene->cameras[cameraId];
				camera->transform = transform;
				process.settings = settings;
				process.context = typename BackwardRenderer<HitType>::
					PixelRenderProcess::Context(scene, camera, frameBuffer);
				int blockStart = (startBlock + (blockIdx.x * blocksPerThread));
				int blockEnd = (blockStart + blocksPerThread);
				if (blockEnd >= endBlock) blockEnd = endBlock;
				process.domain = typename BackwardRenderer<HitType>::PixelRenderProcess::Domain(
					blockStart, blockEnd, blockWidth, blockHeight, threadIdx.x);
				process.resetState();
				//*
				int raycastsLeft = raycastBudget;
				if (!process.render(raycastsLeft)) {
					(*renderEnded) = false;
					//processes[getIndex()] = process;
				}
				/*/
				if (!process.stateActive) return;
				while (true) {
					process.context.frame->setColor(process.posX, process.posY, Color(0, 1, 0));
					if (!process.moveState()) break;
				}
				//(*renderEnded) = false;
				//*/
			}
		}
	}
}
template<typename HitType>
inline void BackwardRenderer<HitType>::iterateGPU(const Info &info) {
	DeviceThreadData &threadData = (*info.getData<DeviceThreadData>());
	bool *renderEndedDevice;
	if (cudaHostGetDevicePointer(&renderEndedDevice, threadData.renderEnded, 0) != cudaSuccess) return;
	Scene<HitType> *scene = config.scene->getHandleGPU(info.device);
	FrameBuffer &cpuBuffer = (*frameBuffer->cpuHandle());
	FrameBuffer *buffer = frameBuffer->gpuHandle(info.device, true);
	if (buffer == NULL) return;
	
	int maxBounce = __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
		::Pixels::getMaxBounces(config.maxBounces);
	typename PixelRenderProcess::Settings settings(
		maxBounce,
		((config.multisampling > 1) ? config.multisampling : 1),
		config.boxingType,
		(1.0f / ((float)iteration())));
	Transform transform = config.scene->getHandleCPU()->cameras[config.cameraId].transform;

	int start, end;
	while (segmentBank.get(threadData.blockCount * config.devicePixelsPerThread, start, end)) {
		(*threadData.renderEnded) = true;
		// __TMP__ __TODO__
		int total = (end - start);
		int numBlocks = ((total > threadData.blockCount) ? threadData.blockCount : total);
		int pixelsPerThread = ((total + numBlocks - 1) / numBlocks);
		
		__DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__::RenderKernels
			::init<HitType><<<numBlocks, threadData.blockSize(), 0, threadData.stream>>>
			(threadData.pixels, settings, scene, config.cameraId, buffer,
				start, end, pixelsPerThread,
				config.segmentWidth, config.segmentHeight, 
				transform, threadData.raycastBudget, renderEndedDevice);
		if (cudaStreamSynchronize(threadData.stream) != cudaSuccess) {
			std::cout << "INIT KERNEL DIED..." << std::endl;
			return;
		}
		else while (!(*threadData.renderEnded)) {
			(*threadData.renderEnded) = true;
			__DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__::RenderKernels
				::render<HitType><<<numBlocks, threadData.blockSize(), 0, threadData.stream>>>
				(threadData.pixels, threadData.raycastBudget, renderEndedDevice);
			if (cudaStreamSynchronize(threadData.stream) != cudaSuccess) {
				std::cout << "RENDER KERNEL DIED..." << std::endl;
				return;
			}
		}
		if(!cpuBuffer.updateHostBlocks(buffer, start, end)) break;
	}
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::completeIteration() {
	if (frameBuffer != NULL) {
		frameBuffer->unlockEdit();
		return true;
	}
	else return false;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::clearData(const Info &info, void *&data) {
	if (data != NULL) {
		if (info.isGPU()) {
			DeviceThreadData *deviceThreadData = ((DeviceThreadData*)data);
			delete deviceThreadData;
			data = NULL;
		}
		else return false;
	}
	return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::clearSharedData(const Info &info, void *&sharedData) {
	return true;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
__dumb__ BackwardRenderer<HitType>::PixelRenderProcess::Settings::Settings(
	int bounce, int samples, BoxingType box, float blend) {
	maxBounces = bounce;
	multisampling = samples;
	boxing = box;
	blendAmount = blend;
}
template<typename HitType>
__dumb__ BackwardRenderer<HitType>::PixelRenderProcess::Context::Context(
	const Scene<HitType> *s, const Camera *c, FrameBuffer *f) {
	scene = s;
	camera = c;
	frame = f;
	if(frame != NULL) frame->getSize(&width, &height);
}
template<typename HitType>
__dumb__ BackwardRenderer<HitType>::PixelRenderProcess::Domain::Domain(
	int startB, int endB, int bWidth, int bHeight, int pixelId) {
	startBlock = startB;
	endBlock = endB;
	blockWidth = bWidth;
	blockHeight = bHeight;
	pixelIndex = pixelId;
}
template<typename HitType>
__dumb__ bool BackwardRenderer<HitType>::PixelRenderProcess::countBase() {
	return __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__::Pixels
		::getPosition(state.block, domain.pixelIndex, domain.blockWidth, domain.blockHeight,
			context.width, context.height, posX, posY);
}
template<typename HitType>
__dumb__ bool BackwardRenderer<HitType>::PixelRenderProcess::moveState() {
	if (state.block < domain.endBlock) {
		while (true) {
			if (state.sampleY < settings.multisampling) {
				if (state.sampleX < settings.multisampling) {
					initPixel(
						posX + (state.sampleX * state.sampleDelta),
						posY + (state.sampleY * state.sampleDelta));
					state.sampleX++;
					stateActive = true;
					return true;
				}
				else state.sampleX = 0;
				state.sampleY++;
			}
			else {
				state.sampleY = 0;
				while (true) {
					state.block++;
					if (state.block >= domain.endBlock) return false;
					if (countBase()) break;
				}
			}
		}
	}
	else return false;
}
template<typename HitType>
__dumb__ void BackwardRenderer<HitType>::PixelRenderProcess::resetState() {
	state.block = domain.startBlock;
	state.sampleX = 0;
	state.sampleY = 0;
	state.sampleDelta = (1.0f / ((float)settings.multisampling));
	state.sampleMass = (state.sampleDelta * state.sampleDelta);
	stateActive = false;
	color = Color(0, 0, 0, 0);
	while (true) {
		if (state.block >= domain.endBlock) return;
		if (countBase()) break;
		state.block++;
	}
}
template<typename HitType>
__dumb__ void BackwardRenderer<HitType>::PixelRenderProcess::initPixel(float x, float y) {
	bounceHeight = 0;
	lightId = 0;
	BounceObject &bounce = bounces[0];
	bounce.color = ColorRGB(0, 0, 0);
	//*
	float divisor;
	if (settings.boxing == VERTICAL) divisor = y;
	else if (settings.boxing == HORIZONTAL) divisor = x;
	else if (settings.boxing == MINIMAL) divisor = ((x < y) ? x : y);
	else if (settings.boxing == MAXIMAL) divisor = ((x > y) ? x : y);
	else divisor = 1;
	float scale = (1.0f / divisor);
	float w = ((float)context.width);
	float h = ((float)context.height);
	screenPoint = Vector2((x - (w / 2.0f)) * scale, ((h / 2.0f) - y) * scale);
	context.camera->getPhotons(screenPoint, bounce.bounces);
	//bounce.bounces.clear();
	//*/
}
template<typename HitType>
__dumb__ bool BackwardRenderer<HitType>::PixelRenderProcess::renderPixel(int &raycastBudget) {
	// __TODO__
	while (true) {
		BounceObject &bounce = bounces[bounceHeight];
		bool isBounce = (!bounce.bounces.empty());
		Photon *photonToCast;
		if (isBounce) photonToCast = (&bounce.bounces.top());
		else if (!lightIllumination.empty()) photonToCast = (&lightIllumination.top());
		else photonToCast = NULL;
		if (photonToCast != NULL) {
			if (raycastBudget <= 0) return false;
			raycastBudget--;
			RaycastHit<Renderable<HitType> > hit;
			if (context.scene->geometry.cast(photonToCast->ray, hit)) {
				if (isBounce) {
					// __PUSH_STACK__
					bounceHeight++;
					BounceObject &topBounce = bounces[bounceHeight];
					topBounce.hit = hit;
					topBounce.bounce = (*photonToCast);
					topBounce.color = ColorRGB(0, 0, 0);
					if (bounceHeight < BACKWARD_RENDERER_MAX_BOUNCES)
						; // __ASK_MATERIAL_FOR_ADDITIONAL_BOUNCE_HERE__
				}
				else if ((bounceHeight > 0) && (hit.object == bounce.hit.object)) {
					// __CALL_SHADER__
				}
			}
			else if ((!isBounce) && (bounceHeight <= 0)) {
				// __CALL_CAMERA__
				Color illumination;
				context.camera->getColor(screenPoint, *photonToCast, illumination);
				bounce.color += illumination;
			}
			if (isBounce) bounce.bounces.pop();
			else lightIllumination.pop();
		}
		else if (lightId < context.scene->lights.size()) {
			// __REQUEST_ILLUMINATION_PHOTONS__
			bool noShadows = false;;
			context.scene->lights[lightId].
				getPhotons(bounce.hit.hitPoint, &noShadows, lightIllumination);
			if (noShadows) {
				if (bounceHeight > 0) {
					for (int i = 0; i < lightIllumination.size(); i++) {
						// __ILLUMIONATE_MATERIAL__
					}
				}
				else {
					// __ILLUMINATE_LENSE__
					for (int i = 0; i < lightIllumination.size(); i++) {
						Color illumination;
						context.camera->getColor(screenPoint, lightIllumination[i], illumination);
						bounce.color += illumination;
					}
				}
				lightIllumination.clear();
			}
			lightId++;
		}
		else {
			lightId = 0;
			// No lights and bounces left;
			if (bounceHeight > 0) {
				// __POP_STACK__
			}
			else {
				// __DONE__ (TODO)
#ifndef __CUDA_ARCH__
				bounce.color = Color(0, 0, 1);
#else
				bounce.color = Color(0, 1, 0);
#endif
				return true;
			}
		}
	}
}
template<typename HitType>
__dumb__ bool BackwardRenderer<HitType>::PixelRenderProcess::render(int &raycastBudget) {
	while (true) {
		if (!stateActive) if (!moveState()) return true;
		if (raycastBudget <= 0) return false;
		if (renderPixel(raycastBudget)) {
			color += (bounces[0].color * state.sampleMass);
			stateActive = false;
			if ((state.sampleX >= settings.multisampling) && 
				(state.sampleY >= (settings.multisampling - 1))) {
				//*
				context.frame->blendColor(posX, posY, color, settings.blendAmount);
				/*/
				context.frame->setColor(posX, posY, color);
				//*/
			}
		}
	}
}





namespace {
	namespace __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__ {
		namespace DeviceDataManagementKernels {
			__dumb__ int blockCount(int multiplier) {
				return (Device::multiprocessorCount() * multiplier);
			}
			__device__ inline int index() { return ((blockIdx.x * blockDim.x) + threadIdx.x); }
			template<typename HitType>
			__global__ static void callPixelRenderProcessConstructors(BackwardRenderer<HitType>::PixelRenderProcess *pixels, int count) {
				int id = index(); if (id < count) new(pixels + id) typename BackwardRenderer<HitType>::PixelRenderProcess();
			}
			template<typename HitType>
			__global__ static void callPixelRenderProcessDestructors(BackwardRenderer<HitType>::PixelRenderProcess *pixels, int count) {
				int id = index(); if (id < count) (pixels + id)->~PixelRenderProcess();
			}
		}
	}
}
template<typename HitType>
inline BackwardRenderer<HitType>::DeviceThreadData::DeviceThreadData(BackwardRenderer *owner) {
	pixels = NULL;
	renderEnded = NULL;
	renderer = owner;
	blockCount = __DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
		::DeviceDataManagementKernels
		::blockCount(renderer->config.deviceOverstauration);
	if (blockCount <= 0) return;
	if (cudaStreamCreate(&stream) != cudaSuccess) return;
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
	__DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
		::DeviceDataManagementKernels
		::callPixelRenderProcessConstructors<HitType>
		<<<blockCount, blockSize(), 0, stream>>>(pixels, pixelCount());
	if (cudaStreamSynchronize(stream) != cudaSuccess) {
		cudaStreamDestroy(stream);
		cudaFreeHost(renderEnded);
		cudaFree(pixels);
		pixels = NULL;
		renderEnded = NULL;
		return;
	}
	raycastBudget = 1;
}
template<typename HitType>
inline BackwardRenderer<HitType>::DeviceThreadData::~DeviceThreadData() {
	bool streamExists = false;
	if (pixels != NULL) {
		__DumbRay_BACKWARD_RENDERER_PRIVATE_NAMESPACE__
			::DeviceDataManagementKernels
			::callPixelRenderProcessDestructors<HitType>
			<<<blockCount, blockSize(), 0, stream>>>(pixels, pixelCount());
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
	return (blockCount * blockSize());
}
template<typename HitType>
inline int BackwardRenderer<HitType>::DeviceThreadData::blockSize()const {
	return (renderer->config.segmentWidth * renderer->config.segmentHeight);
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

