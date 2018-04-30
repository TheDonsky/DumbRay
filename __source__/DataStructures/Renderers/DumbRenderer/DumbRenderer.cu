#include "DumbRenderer.cuh"



DumbRenderer::DumbRenderer(
	const ThreadConfiguration &configuration,
	const BlockConfiguration &blockSettings,
	FrameBufferManager *buffer,
	SceneType *scene,
	CameraManager *camera,
	BoxingMode boxingMode)
	: BlockRenderer(configuration, blockSettings, buffer) {
	setScene(scene);
	setCamera(camera);
	setBoxingMode(boxing);
}

void DumbRenderer::setScene(SceneType *scene) { sceneManager = scene; }
DumbRenderer::SceneType* DumbRenderer::getScene()const { return (SceneType*)sceneManager; }

void DumbRenderer::setCamera(CameraManager *camera) { cameraManager = camera; }
DumbRenderer::CameraManager* DumbRenderer::getCamera()const { return (CameraManager*)cameraManager; }

void DumbRenderer::setBoxingMode(BoxingMode mode) { boxing = mode; }
DumbRenderer::BoxingMode DumbRenderer::getBoxingMode()const { return boxing; }


bool DumbRenderer::renderBlocksCPU(
	const Info &info, FrameBuffer *buffer, int startBlock, int endBlock) {
	PixelRenderProcess::SceneConfiguration configuration;
	float blending = ((iteration() <= 1) ? 1.0f : (1.0f / ((float)iteration())));
	if (!configuration.host(getScene(), getCamera(), buffer, boxing, blending)) return false;
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
	if (!configuration.device(getScene(), getCamera(), device, host, boxing, info.device, blending)) return false;
	DumbRendererPrivateKernels::renderBlocks
		<<<(endBlock - startBlock), host->getBlockSize(), 0, renderStream>>>
		(configuration, startBlock);
	if (cudaStreamSynchronize(renderStream) != cudaSuccess)
		printf("error: %d\n", (int)cudaGetLastError());
	return true;
}


bool DumbRenderer::PixelRenderProcess::SceneConfiguration::host(
	SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, BoxingMode boxingMode, float blending) {
	context.host(scene);
	camera = cameraManager->cpuHandle();
	buffer = frameBuffer;
	boxing = boxingMode;
	frameBuffer->getSize(&width, &height);
	blendingAmount = blending;
	return (!hasError());
}
bool DumbRenderer::PixelRenderProcess::SceneConfiguration::device(
	SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, FrameBuffer *hostFrameBuffer, BoxingMode boxingMode, int deviceId, float blending) {
	context.device(scene, deviceId);
	camera = cameraManager->gpuHandle(deviceId);
	buffer = frameBuffer;
	boxing = boxingMode;
	hostFrameBuffer->getSize(&width, &height);
	blendingAmount = blending;
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
		//pixelX = 512;
		//pixelY = 512;
		return true;
	}
	else return false;
}

__device__ __host__ void DumbRenderer::PixelRenderProcess::render() {
	// __TMP__:

	// Relative pixel location and size:
	//*
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
	RaySamples samples;
	//*
	configuration.camera->getPixelSamples(offset, pixelSize, samples);
	if (samples.sampleCount != 1) return;
	/*/
	samples.sampleCount = 1;
	samples.samples[0] = SampleRay(Ray(Vector3(0, 0, -128), Vector3(offset.x, offset.y, 1).normalized()), 1);
	//*/
	//*/
	Color color = Color(0.0f, 0.0f, 0.0f, 1.0f);
	for (int i = 0; i < samples.sampleCount; i++) {
		RaycastHit<SceneType::GeometryUnit> hit;
		if (configuration.context.geometry->cast(samples.samples[i].ray, hit, false)) {
#ifdef __CUDA_ARCH__
			Color col(0, 1, 0, 1);
#else
			Color col(0, 0, 1, 1);
#endif
			color += (col * samples.samples[i].sampleWeight);
		}
	}
	/*/

#ifdef __CUDA_ARCH__
	Color color(0, 1, 0, 1);
#else
	Color color(0, 0, 1, 1);
#endif
	//*/
	//*
	if (configuration.blendingAmount >= 1.0f)
		configuration.buffer->setBlockPixelColor(block, pixelInBlock, color);
	else configuration.buffer->blendBlockPixelColor(block, pixelInBlock, color, configuration.blendingAmount);
	//*/
}
