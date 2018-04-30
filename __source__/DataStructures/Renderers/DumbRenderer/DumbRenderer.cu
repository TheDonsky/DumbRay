#include "DumbRenderer.cuh"


#define DUMB_RENDERER_BOUNCE_LIMIT 128


DumbRenderer::DumbRenderer(
	const ThreadConfiguration &configuration,
	const BlockConfiguration &blockSettings,
	FrameBufferManager *buffer,
	SceneType *scene,
	CameraManager *camera,
	BoxingMode boxingMode,
	int maxBounces)
	: BlockRenderer(configuration, blockSettings, buffer) {
	setScene(scene);
	setCamera(camera);
	setBoxingMode(boxing);
	setMaxBounces(maxBounces);
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

bool DumbRenderer::renderBlocksCPU(
	const Info &info, FrameBuffer *buffer, int startBlock, int endBlock) {
	PixelRenderProcess::SceneConfiguration configuration;
	float blending = ((iteration() <= 1) ? 1.0f : (1.0f / ((float)iteration())));
	if (!configuration.host(getScene(), getCamera(), buffer, boxing, blending, getMaxBounces())) return false;
	PixelRenderProcess pixelRenderProcess;
	pixelRenderProcess.configure(configuration);
	int blockSize = buffer->getBlockSize();
	for (int blockId = startBlock; blockId < endBlock; blockId++)
		for (int pixelId = 0; pixelId < blockSize; pixelId++) {
			if (!pixelRenderProcess.setPixel(blockId, pixelId)) continue;
			pixelRenderProcess.render();
		}
	return true;
}

namespace {
	namespace DumbRendererPrivateKernels {
		__global__ static void renderBlocks(
			DumbRenderer::PixelRenderProcess::SceneConfiguration configuration, int startBlock) {
			DumbRenderer::PixelRenderProcess pixelRenderProcess;
			pixelRenderProcess.configure(configuration);
			if (!pixelRenderProcess.setPixel(startBlock + blockIdx.x, threadIdx.x)) return;
			pixelRenderProcess.render();
		}
	}
}

bool DumbRenderer::renderBlocksGPU(
	const Info &info, FrameBuffer *host, FrameBuffer *device, 
	int startBlock, int endBlock, cudaStream_t &renderStream) {
	PixelRenderProcess::SceneConfiguration configuration;
	float blending = ((iteration() <= 1) ? 1.0f : (1.0f / ((float)iteration())));
	if (!configuration.device(getScene(), getCamera(), device, host, boxing, info.device, blending, getMaxBounces())) return false;
	DumbRendererPrivateKernels::renderBlocks
		<<<(endBlock - startBlock), host->getBlockSize(), 0, renderStream>>>
		(configuration, startBlock);
	if (cudaStreamSynchronize(renderStream) != cudaSuccess)
		printf("error: %d\n", (int)cudaGetLastError());
	return true;
}


bool DumbRenderer::PixelRenderProcess::SceneConfiguration::host(
	SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, BoxingMode boxingMode, float blending, int bounces) {
	context.host(scene);
	camera = cameraManager->cpuHandle();
	buffer = frameBuffer;
	boxing = boxingMode;
	frameBuffer->getSize(&width, &height);
	blendingAmount = blending;
	maxBounces = bounces;
	return (!hasError());
}
bool DumbRenderer::PixelRenderProcess::SceneConfiguration::device(
	SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, FrameBuffer *hostFrameBuffer, BoxingMode boxingMode, int deviceId, float blending, int bounces) {
	context.device(scene, deviceId);
	camera = cameraManager->gpuHandle(deviceId);
	buffer = frameBuffer;
	boxing = boxingMode;
	hostFrameBuffer->getSize(&width, &height);
	blendingAmount = blending;
	maxBounces = bounces;
	return (!hasError());
}
bool DumbRenderer::PixelRenderProcess::SceneConfiguration::hasError() {
	return (context.hasError() || (camera == NULL) || (buffer == NULL));
}

__device__ __host__ void DumbRenderer::PixelRenderProcess::configure(const SceneConfiguration &config) {
	configuration = config;
}
__device__ __host__ bool DumbRenderer::PixelRenderProcess::setPixel(int blockId, int pixelId) {
	if (configuration.buffer->blockPixelLocation(blockId, pixelId, &pixelX, &pixelY)) {
		block = blockId;
		pixelInBlock = pixelId;
		return true;
	}
	else return false;
}

__device__ __host__ void DumbRenderer::PixelRenderProcess::render() {
	// Relative pixel location and size:
	register BoxingMode boxing = configuration.boxing;
	register float width = configuration.width;
	register float height = configuration.height;
	float pixelSize;
	if (boxing == BOXING_MODE_HEIGHT_BASED) pixelSize = (1.0f / height);
	else if (boxing == BOXING_MODE_WIDTH_BASED) pixelSize = (1.0f / width);
	else if (boxing == BOXING_MODE_MIN_BASED) pixelSize = (1.0f / ((height <= width) ? height : width));
	else if (boxing == BOXING_MODE_MAX_BASED) pixelSize = (1.0f / ((height >= width) ? height : width));
	else pixelSize = 1.0f;
	Vector2 offset((pixelX - (width / 2.0f)) * pixelSize, ((height / 2.0f) - pixelY) * pixelSize);
	RaySamples cameraPixelSamples;
	configuration.camera->getPixelSamples(offset, pixelSize, cameraPixelSamples);
	Color color = Color(0.0f, 0.0f, 0.0f, 0.0f);
	//*
	BounceLayer bounceLayers[DUMB_RENDERER_BOUNCE_LIMIT];
	PhotonSamples lightRays;
	int currentLayer = -1;
	int maxLayer = configuration.maxBounces;
	while (true) {
		if (currentLayer < 0) {
			if (cameraPixelSamples.sampleCount <= 0) break;
			cameraPixelSamples.sampleCount--;
			currentLayer = 0;
			bounceLayers[currentLayer].setup(
				cameraPixelSamples.samples[cameraPixelSamples.sampleCount], 1.0f);
			lightRays.sampleCount = 0;
		}
		BounceLayer &layer = bounceLayers[currentLayer];
		
		const Ray *rayToCast;
		bool isLightRay = false;
		if (layer.geometry.object == NULL) rayToCast = (&layer.layerRay);
		else if (lightRays.sampleCount > 0) {
			lightRays.sampleCount--;
			rayToCast = &lightRays.samples[lightRays.sampleCount].ray;
		}
		else if (layer.lightIndex < configuration.context.lights->size()) {
			// __TODO__: add light samples here...
			layer.lightIndex++;
			continue;
		}
		else if (layer.bounces.sampleCount > 0) {
			layer.bounces.sampleCount--;
			currentLayer++;
			bounceLayers[currentLayer].setup(
				layer.bounces.samples[layer.bounces.sampleCount], layer.absoluteWeight);
			lightRays.sampleCount = 0;
			continue;
		}
		else {
			if (currentLayer > 0) {
				// __TODO__: illuminate the underlying geometry...
			}
			else {
				// __TODO__: add color to the final pixel...
			}
			currentLayer--;
			continue;
		}
		RaycastHit<SceneType::GeometryUnit> hit;
		if (configuration.context.geometry->cast(*rayToCast, hit, false)) {
			if (rayToCast == (&layer.layerRay)) {
				layer.geometry = hit;
				if (currentLayer < maxLayer) {
					// __TODO__: Request for bounces...
				}

				// TMP (HAS TO BE REMOVED...):
#ifdef __CUDA_ARCH__
				Color col(0, 1, 0, 1);
#else
				Color col(0, 0, 1, 1);
#endif
				color += (col * layer.sampleWeight);
				continue;
			}
			else {
				// __TODO__: illuminate the pixel...
			}
		}
		else if (rayToCast == (&layer.layerRay)) {
			currentLayer--;
			// __TODO__: MAYBE... Nothing?
			continue;
		}
		else {
			// __TODO__: Probably nothing...
			continue;
		}
	}
	/*/
	for (int i = 0; i < cameraPixelSamples.sampleCount; i++) {
		RaycastHit<SceneType::GeometryUnit> hit;
		if (configuration.context.geometry->cast(cameraPixelSamples.samples[i].ray, hit, false)) {
#ifdef __CUDA_ARCH__
			Color col(0, 1, 0, 1);
#else
			Color col(0, 0, 1, 1);
#endif
			color += (col * cameraPixelSamples.samples[i].sampleWeight);
		}
	}
	//*/
	
	if (configuration.blendingAmount >= 1.0f)
		configuration.buffer->setBlockPixelColor(block, pixelInBlock, color);
	else configuration.buffer->blendBlockPixelColor(block, pixelInBlock, color, configuration.blendingAmount);
}
