#include "DumbRenderer.cuh"



DumbRenderer::DumbRenderer(
	const ThreadConfiguration &configuration,
	const BlockConfiguration &blockSettings,
	FrameBufferManager *buffer,
	SceneType *scene,
	CameraManager *camera) 
	: BlockRenderer(configuration, blockSettings, buffer) {
	setScene(scene);
	setCamera(camera);
}

void DumbRenderer::setScene(SceneType *scene) { sceneManager = scene; }
DumbRenderer::SceneType* DumbRenderer::getScene()const { return sceneManager; }

void DumbRenderer::setCamera(CameraManager *camera) { cameraManager = camera; }
DumbRenderer::CameraManager* DumbRenderer::getCamera()const { return cameraManager; }



bool DumbRenderer::renderBlocksCPU(
	const Info &info, FrameBuffer *buffer, int startBlock, int endBlock) {
	PixelRenderProcess::SceneConfiguration configuration;
	if (!configuration.host(getScene(), getCamera(), buffer)) return false;
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
	if (!configuration.device(getScene(), getCamera(), device, info.device)) return false;
	DumbRendererPrivateKernels::renderBlocks
		<<<(endBlock - startBlock), host->getBlockSize(), 0, renderStream>>>
		(configuration, startBlock);
	return true;
}


bool DumbRenderer::PixelRenderProcess::SceneConfiguration::host(
	SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer) {
	context.host(scene);
	camera = cameraManager->cpuHandle();
	buffer = frameBuffer;
	return (!hasError());
}
bool DumbRenderer::PixelRenderProcess::SceneConfiguration::device(
	SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, int deviceId) {
	context.device(scene, deviceId);
	camera = cameraManager->gpuHandle(deviceId);
	buffer = frameBuffer;
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
		reset(); return true;
	}
	else return false;
}
__device__ __host__ void DumbRenderer::PixelRenderProcess::reset() {
	// __TODO__: 
}

__device__ __host__ void DumbRenderer::PixelRenderProcess::render() {
	// __TMP__:
#ifdef __CUDA_ARCH__
	Color color(0, 1, 0, 1);
#else
	Color color(0, 0, 1, 1);
#endif
	configuration.buffer->setBlockPixelColor(block, pixelInBlock, color);
}
