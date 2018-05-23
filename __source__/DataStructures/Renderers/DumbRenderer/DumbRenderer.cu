#include "DumbRenderer.cuh"


DumbRenderer::DumbRenderer(
	const ThreadConfiguration &configuration,
	const BlockConfiguration &blockSettings,
	FrameBufferManager *buffer,
	SceneType *scene,
	CameraManager *camera,
	BoxingMode boxingMode,
	int maxBounces,
	int samplesPerPixelX,
	int samplesPerPixelY,
	int pixelsPerGPUThread)
	: BlockRenderer(configuration, blockSettings, buffer) {
	setScene(scene);
	setCamera(camera);
	setBoxingMode(boxingMode);
	setMaxBounces(maxBounces);
	setSamplesPerPixel(samplesPerPixelX, samplesPerPixelY);
	setPixelsPerGPUThread(pixelsPerGPUThread);
}

void DumbRenderer::setScene(SceneType *scene) { sceneManager = scene; }
DumbRenderer::SceneType* DumbRenderer::getScene()const { return (SceneType*)sceneManager; }

void DumbRenderer::setCamera(CameraManager *camera) { cameraManager = camera; }
DumbRenderer::CameraManager* DumbRenderer::getCamera()const { return (CameraManager*)cameraManager; }

void DumbRenderer::setBoxingMode(BoxingMode mode) { boxing = mode; }
DumbRenderer::BoxingMode DumbRenderer::getBoxingMode()const { return boxing; }

void DumbRenderer::setMaxBounces(int maxBounces) { bounceLimit = ((maxBounces < maxBouncesLimit()) ? maxBounces : maxBouncesLimit()); }
int DumbRenderer::getMaxBounces()const { return bounceLimit; }

int DumbRenderer::maxBouncesLimit() { return DUMB_RENDERER_BOUNCE_LIMIT; }

void DumbRenderer::setSamplesPerPixelX(int value) { fsaaX = ((value <= 1) ? 1 : value); }
void DumbRenderer::setSamplesPerPixelY(int value) { fsaaY = ((value <= 1) ? 1 : value); }
void DumbRenderer::setSamplesPerPixel(int x, int y) { setSamplesPerPixelX(x); setSamplesPerPixelY(y); }
int DumbRenderer::getSamplesPerPixelX()const { return fsaaX; }
int DumbRenderer::getSamplesPerPixelY()const { return fsaaY; }

void DumbRenderer::setPixelsPerGPUThread(int count) { pxPerGPUThread = count; }
int DumbRenderer::getPixelsPerGPUThread()const { return pxPerGPUThread; }

bool DumbRenderer::renderBlocksCPU(
	const Info &info, FrameBuffer *buffer, int startBlock, int endBlock) {
	DumbRandHolder *randHolder = getThreadData<DumbRandHolder>(info);
	if (randHolder == NULL) return false;
	DumbRand *entropy = randHolder->getCPU(1, false);
	if (entropy == NULL) return false;
	RenderContext renderContext;
	renderContext.entropy = entropy;

	PixelRenderProcess::SceneConfiguration sceneConfiguration;
	float blending = ((iteration() <= 1) ? 1.0f : (1.0f / ((float)iteration())));
	if (!sceneConfiguration.host(getScene(), getCamera(), buffer, boxing, blending, getMaxBounces(), fsaaX, fsaaY)) return false;
	PixelRenderProcess pixelRenderProcess;
	pixelRenderProcess.configure(sceneConfiguration);
	pixelRenderProcess.setContext(renderContext, 0);
	int blockSize = buffer->getBlockSize();
	for (int pixelId = 0; pixelId < blockSize; pixelId++)
		pixelRenderProcess.renderPixels(pixelId, startBlock, endBlock, 1);
	return true;
}

namespace {
	namespace DumbRendererPrivateKernels {
		__global__ static void renderBlocks(
			DumbRenderer::PixelRenderProcess::SceneConfiguration configuration, RenderContext renderContext, int startBlock, int endBlock) {
			DumbRenderer::PixelRenderProcess pixelRenderProcess;
			pixelRenderProcess.configure(configuration);
			pixelRenderProcess.setContext(renderContext, (blockIdx.x * blockDim.x) + threadIdx.x);
			pixelRenderProcess.renderPixels(threadIdx.x, (startBlock + blockIdx.x), endBlock, gridDim.x);
		}
	}
}

bool DumbRenderer::renderBlocksGPU(
	const Info &info, FrameBuffer *host, FrameBuffer *device, 
	int startBlock, int endBlock, cudaStream_t &renderStream) {
	DumbRandHolder *randHolder = getThreadData<DumbRandHolder>(info);
	if (randHolder == NULL) return false;

	const int blocksToRender = (endBlock - startBlock);
	const int blockCount = ((blocksToRender + pxPerGPUThread - 1) / pxPerGPUThread);

	DumbRand *entropy = randHolder->getGPU(blockCount * host->getBlockSize(), info.device, false);
	if (entropy == NULL) return false;
	RenderContext renderContext;
	renderContext.entropy = entropy;

	PixelRenderProcess::SceneConfiguration sceneConfiguration;
	float blending = ((iteration() <= 1) ? 1.0f : (1.0f / ((float)iteration())));
	if (!sceneConfiguration.device(getScene(), getCamera(), device, host, boxing, info.device, blending, getMaxBounces(), fsaaX, fsaaY)) return false;
	DumbRendererPrivateKernels::renderBlocks
		<<<blockCount, host->getBlockSize(), 0, renderStream>>>
		(sceneConfiguration, renderContext, startBlock, endBlock);
	if (cudaStreamSynchronize(renderStream) != cudaSuccess)
		printf("error: %d\n", (int)cudaGetLastError());
	return true;
}


bool DumbRenderer::PixelRenderProcess::SceneConfiguration::host(
	SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, 
	BoxingMode boxingMode, float blending, int bounces, int samplesX, int samplesY) {
	context.host(scene);
	camera = cameraManager->cpuHandle();
	buffer = frameBuffer;
	boxing = boxingMode;
	frameBuffer->getSize(&width, &height);
	blendingAmount = blending;
	maxBounces = bounces;
	fsaaX = samplesX;
	fsaaY = samplesY;
	return (!hasError());
}
bool DumbRenderer::PixelRenderProcess::SceneConfiguration::device(
	SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, FrameBuffer *hostFrameBuffer, 
	BoxingMode boxingMode, int deviceId, float blending, int bounces, int samplesX, int samplesY) {
	context.device(scene, deviceId);
	camera = cameraManager->gpuHandle(deviceId);
	buffer = frameBuffer;
	boxing = boxingMode;
	hostFrameBuffer->getSize(&width, &height);
	blendingAmount = blending;
	maxBounces = bounces;
	fsaaX = samplesX;
	fsaaY = samplesY;
	return (!hasError());
}
bool DumbRenderer::PixelRenderProcess::SceneConfiguration::hasError() {
	return (context.hasError() || (camera == NULL) || (buffer == NULL));
}

__device__ __host__ void DumbRenderer::PixelRenderProcess::configure(const SceneConfiguration &config) {
	configuration = config;
}

__device__ __host__ void DumbRenderer::PixelRenderProcess::setContext(const RenderContext &context, int entropyOffset) {
	renderContext = context;
	if (renderContext.entropy != NULL)
		renderContext.entropy += entropyOffset;
}

__device__ __host__ void DumbRenderer::PixelRenderProcess::BounceLayer::setup(const SampleRay &sample, float absWeight) {
	color = Color(0.0f, 0.0f, 0.0f, 0.0f);
	layerRay = sample.ray;
	sampleWeight = sample.sampleWeight;
	sampleSignificance = sample.significance;
	sampleType = sample.type;
	absoluteWeight = sampleWeight * absWeight;
	geometry.object = NULL;
	lightIndex = 0;
	bounces.sampleCount = 0;
}



__device__ __host__ void DumbRenderer::PixelRenderProcess::renderPixels(int pixelId, int startBlock, int endBlock, int step) {
	pixelInBlock = pixelId;
	curBlock = startBlock;
	stopBlock = endBlock;
	blockStep = step;
	maxLayer = configuration.maxBounces;
	countPixelSize();	
	if (!setPixel()) return;
	if (!setSubPixel()) return;
	while (true)
		if (subPixelRenderPass())
			if (!setSubPixel()) {
				if (!setPixel()) break;
				if (!setSubPixel()) return;
			}
}
__device__ __host__ void DumbRenderer::PixelRenderProcess::countPixelSize() {
	register float width = (float)configuration.width;
	register float height = (float)configuration.height;
	register BoxingMode boxingMode = configuration.boxing;
	if (boxingMode == BOXING_MODE_HEIGHT_BASED) pixelSize = (1.0f / height);
	else if (boxingMode == BOXING_MODE_WIDTH_BASED) pixelSize = (1.0f / width);
	else if (boxingMode == BOXING_MODE_MIN_BASED) pixelSize = (1.0f / ((height <= width) ? height : width));
	else if (boxingMode == BOXING_MODE_MAX_BASED) pixelSize = (1.0f / ((height >= width) ? height : width));
	else pixelSize = 1.0f;
}
__device__ __host__ bool DumbRenderer::PixelRenderProcess::setPixel() {
	if (curBlock >= stopBlock) return false;
	{
		int pixelX, pixelY;
		register float width = (float)configuration.width;
		register float height = (float)configuration.height;
		if (!configuration.buffer->blockPixelLocation(curBlock, pixelInBlock, &pixelX, &pixelY)) return false;
		pixelPostion = Vector2((pixelX - (width / 2.0f)) * pixelSize, ((height / 2.0f) - pixelY) * pixelSize);
		pixelColor = Color(0.0f, 0.0f, 0.0f, 0.0f);
	}
	{
		fsaaI = fsaaJ = 0;
		subPixelColor = Color(0.0f, 0.0f, 0.0f, 0.0f);
		subPixelWeight = 0.0f;
	}
	return true;
}
__device__ __host__ bool DumbRenderer::PixelRenderProcess::setSubPixel() {
	pixelColor += (subPixelColor * subPixelWeight);
	if (fsaaI >= configuration.fsaaX) {
		if (configuration.blendingAmount >= 1.0f)
			configuration.buffer->setBlockPixelColor(curBlock, pixelInBlock, pixelColor);
		else configuration.buffer->blendBlockPixelColor(curBlock, pixelInBlock, pixelColor, configuration.blendingAmount);
		curBlock += blockStep;
		return false;
	}
	else {
		{
			Vector2 sampleOffset(
				(((float)fsaaI) + 0.5f) * pixelSize / ((float)configuration.fsaaX),
				(((float)fsaaJ) + 0.5f) *  pixelSize / ((float)configuration.fsaaY));
			screenSpacePosition = (pixelPostion + sampleOffset);
			LenseGetPixelSamplesRequest cameraPixelSamplesRequest;
			cameraPixelSamplesRequest.screenSpacePosition = screenSpacePosition;
			cameraPixelSamplesRequest.pixelSize = pixelSize;
			cameraPixelSamplesRequest.context = (&renderContext);
			configuration.camera->getPixelSamples(cameraPixelSamplesRequest, &cameraPixelSamples);
		}
		{
			subPixelColor = Color(0.0f, 0.0f, 0.0f, 0.0f);
			subPixelWeight = (1.0f / (((float)configuration.fsaaX) * ((float)configuration.fsaaY)));
			fsaaJ++; if (fsaaJ >= configuration.fsaaY) { fsaaJ = 0; fsaaI++; }
			currentLayer = -1;
		}
		return true;
	}
}
__device__ __host__ bool DumbRenderer::PixelRenderProcess::subPixelRenderPass() {
	if (!setupSubPixelRenderPassLayer()) return true;
	if (setSubPixelRenderPassJob())
		if (castSubPixelRenderPassRay())
			illuminateSubPixelRenderPassLayer();
	return false;
}
__device__ __host__ bool DumbRenderer::PixelRenderProcess::setupSubPixelRenderPassLayer() {
	// If current layer is 'negative', it means that the render has not started yet, 
	// or it ended, or there are still some camera samples left to investigate:
	if (currentLayer < 0) {
		// If there are no more camera samples, we end the render process:
		if (cameraPixelSamples.sampleCount <= 0) return false;

		// Else, we set the first layer up with a camera sample and decrease the counter:
		cameraPixelSamples.sampleCount--;
		currentLayer = 0;
		bounceLayers[currentLayer].setup(
			cameraPixelSamples.samples[cameraPixelSamples.sampleCount], 1.0f);
		lightRays.sampleCount = 0;
	}

	// For convenience, current bounce layer will be reffered to as 'layer':
	layer = (bounceLayers + currentLayer);
	return true;
}
__device__ __host__ bool DumbRenderer::PixelRenderProcess::setSubPixelRenderPassJob() {
	// We may need to cast the ray either to investigate the direction of the layer,
	// or illuminate it. Regardless, we'll need a reference to a ray to cast (this way we decrease diverency):
	restrictionObject = NULL;

	// We may need to call the shader's getReflectedColor for aome amount of photons:
	numIlluminationPhotons = 0;

	// In case there's no geometry set, rayToCast is the layer ray:
	if (layer->geometry.object == NULL) {
		rayToCast = (&layer->layerRay);
		if (currentLayer > 0) restrictionObject = ((void*)bounceLayers[currentLayer - 1].geometry.object);
	}

	// If geometry is already set and we have uncast photons, 
	// we will simply cast one of them and decrease the counter:
	else if (lightRays.sampleCount > 0) {
		lightRays.sampleCount--;
		rayToCast = &lightRays.samples[lightRays.sampleCount].ray;
	}


	// ----------------------------------------------

	// If we came through here without an obvious ray to cast, 
	// we try to extract more light samples:
	else if (layer->lightIndex < configuration.context.lights->size()) {
		// __TODO__: add light samples here...
		bool castShadows = true;
		LightVertexSampleRequest request;
		request.point = layer->geometry.hitPoint;
		request.context = (&renderContext);
		configuration.context.lights->operator[](layer->lightIndex).getVertexPhotons(
			request, &lightRays, &castShadows);
		layer->lightIndex++;

		if (castShadows) return false;
		else {
			numIlluminationPhotons = lightRays.sampleCount;
			lightRays.sampleCount = 0;
			rayToCast = NULL;
		}
	}

	// With geometry already investigated and no direct illumination photons left,
	// all we have to do is to either go on and setup the indirect sample if it's still pending:
	else if (layer->bounces.sampleCount > 0) {
		layer->bounces.sampleCount--;
		currentLayer++;
		bounceLayers[currentLayer].setup(
			layer->bounces.samples[layer->bounces.sampleCount], layer->absoluteWeight);
		lightRays.sampleCount = 0;
		return false;
	}

	// In case, there are no more indirect samples requested, all can do is to go down a layer:
	else {
		if (currentLayer > 0) {
			// Illuminate the underlying geometry:
			BounceLayer &layerBelow = bounceLayers[currentLayer - 1];
			ShaderReflectedColorRequest<SceneType::SurfaceUnit> request;
			request.object = &layerBelow.geometry.object->object;
			request.photon = Photon(Ray(layerBelow.geometry.hitPoint, -layer->layerRay.direction), layer->color);
			request.hitPoint = layerBelow.geometry.hitPoint;
			request.observerDirection = (-layerBelow.layerRay.direction);
			request.photonType = PHOTON_TYPE_INDIRECT_ILLUMINATION;
			request.significance = layer->sampleSignificance;
			request.sampleType = layer->sampleType;
			request.context = (&renderContext);
			layerBelow.color += (configuration.context.materials->operator[](
				layerBelow.geometry.object->materialId).getReflectedColor(request) * layer->sampleWeight);
		}
		else {
			// Add color to the final pixel:
			LenseGetPixelColorRequest request;
			request.screenSpacePosition = screenSpacePosition;
			request.pixelSize = pixelSize;
			request.photon = Photon(Ray(layer->geometry.hitPoint, -layer->layerRay.direction), layer->color);
			request.photonType = PHOTON_TYPE_INDIRECT_ILLUMINATION;
			request.context = (&renderContext);
			subPixelColor += (configuration.camera->getPixelColor(request) * layer->sampleWeight);
		}
		currentLayer--;
		return false;
	}
	return true;
}
__device__ __host__ bool DumbRenderer::PixelRenderProcess::castSubPixelRenderPassRay() {
	if (rayToCast != NULL) {
		// If we have a ray to cast, we should be here at some point:
		RaycastHit<SceneType::GeometryUnit> hit;
		if (configuration.context.geometry->cast(
			*rayToCast, hit, false, Octree<SceneType::GeometryUnit>::validateNotSameAsObject, restrictionObject)) {
			// If raycast hit something and it was a geometry ray,
			// we need to set the layer up:
			if (rayToCast == (&layer->layerRay)) {
				layer->geometry = hit;
				// If we are allawed to further request indirect illumination,
				// here's the time:
				if (currentLayer < maxLayer) {
					ShaderIndirectSamplesRequest<SceneType::SurfaceUnit> request;
					request.absoluteSampleWeight = layer->absoluteWeight;
					request.hitDistance = hit.hitDistance;
					request.hitPoint = hit.hitPoint;
					request.object = &hit.object->object;
					request.ray = (*rayToCast);
					request.significance = layer->sampleSignificance;
					request.sampleType = layer->sampleType;
					request.context = (&renderContext);
					configuration.context.materials->operator[](
						hit.object->materialId).requestIndirectSamples(request, &layer->bounces);
				}
				return false;
			}
			// If the ray was an illumination ray and it hit the same old object or point, we count it as a light:
			else if (
				(hit.object == layer->geometry.object) ||
				((hit.hitPoint - layer->geometry.hitPoint).sqrMagnitude() <= (8.0f * VECTOR_EPSILON))) {
				numIlluminationPhotons = 1;
			}
			// If no illumination occured whatsoever, there's no point continuing the cycle:
			else return false;
		}
		// If raycast failed and it was a geometry ray, we have to give up on the current layer:
		else if (rayToCast == (&layer->layerRay)) {
			currentLayer--;
			return false;
		}
		// If raycast failed and it was a light ray, we probably don't care about it all that much,
		// even though ity should never happen in theory:
		else return false;
	}
	return true;
}
__device__ __host__ void DumbRenderer::PixelRenderProcess::illuminateSubPixelRenderPassLayer() {
	const SceneType::MaterialType &layerMaterial = configuration.context.materials->operator[](layer->geometry.object->materialId);
	for (int i = 0; i < numIlluminationPhotons; i++) {
		ShaderReflectedColorRequest<SceneType::SurfaceUnit> request;
		request.object = &layer->geometry.object->object;
		request.photon = lightRays.samples[lightRays.sampleCount + i];
		request.hitPoint = layer->geometry.hitPoint;
		request.observerDirection = (-layer->layerRay.direction);
		request.photonType = PHOTON_TYPE_DIRECT_ILLUMINATION;
		request.significance = 1.0f;
		request.sampleType = 0;
		request.context = (&renderContext);
		layer->color += layerMaterial.getReflectedColor(request);
	}
}
