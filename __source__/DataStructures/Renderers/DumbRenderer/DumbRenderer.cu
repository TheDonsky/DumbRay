#include "DumbRenderer.cuh"


#define DUMB_RENDERER_BOUNCE_LIMIT 64


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
	for (int blockId = startBlock; blockId < endBlock; blockId++)
		for (int pixelId = 0; pixelId < blockSize; pixelId++)
			pixelRenderProcess.renderPixel(blockId, pixelId);
	return true;
}

namespace {
	namespace DumbRendererPrivateKernels {
		__global__ static void renderBlocks(
			DumbRenderer::PixelRenderProcess::SceneConfiguration configuration, RenderContext renderContext, int startBlock, int endBlock) {
			DumbRenderer::PixelRenderProcess pixelRenderProcess;
			pixelRenderProcess.configure(configuration);
			pixelRenderProcess.setContext(renderContext, (blockIdx.x * blockDim.x) + threadIdx.x);
			for (int curBlock = (startBlock + blockIdx.x); curBlock < endBlock; curBlock += gridDim.x)
				pixelRenderProcess.renderPixel(curBlock, threadIdx.x);
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

__device__ __host__ void DumbRenderer::PixelRenderProcess::renderPixel(int blockId, int pixelId) {
	// ###################################################################
	/* Relative pixel size: */

	// Extructing some parameters:
	register float width = (float)configuration.width;
	register float height = (float)configuration.height;

	// Pixel size (relative to lense):
	float pixelSize;
	{
		register BoxingMode boxing = configuration.boxing;
		if (boxing == BOXING_MODE_HEIGHT_BASED) pixelSize = (1.0f / height);
		else if (boxing == BOXING_MODE_WIDTH_BASED) pixelSize = (1.0f / width);
		else if (boxing == BOXING_MODE_MIN_BASED) pixelSize = (1.0f / ((height <= width) ? height : width));
		else if (boxing == BOXING_MODE_MAX_BASED) pixelSize = (1.0f / ((height >= width) ? height : width));
		else pixelSize = 1.0f;
	}



	// ###################################################################
	/* Pixel location detection: */

	// Pixel position (Relative to lense center):
	Vector2 offset;
	{
		int pixelX, pixelY;
		if (!configuration.buffer->blockPixelLocation(blockId, pixelId, &pixelX, &pixelY)) return;
		offset = Vector2((pixelX - (width / 2.0f)) * pixelSize, ((height / 2.0f) - pixelY) * pixelSize);
	}




	// ###################################################################
	/* Ray tracing logic: */
	Color result(0.0f, 0.0f, 0.0f, 0.0f);

	int fsaaI = configuration.fsaaX;
	int fsaaJ = configuration.fsaaY;

	const float pixelSampleW = (1.0f / ((float)configuration.fsaaX));
	const float pixelSampleH = (1.0f / ((float)configuration.fsaaX));
	Vector2 screenSpacePosition;

	// Layers for each bounce:
	BounceLayer bounceLayers[DUMB_RENDERER_BOUNCE_LIMIT + 1];
	PhotonSamples lightRays;
	const int maxLayer = configuration.maxBounces;

	// __________________________________________________
	// Full screen anti aliasing loop:
	for (fsaaI = 0; fsaaI < configuration.fsaaX; fsaaI++)
		for (fsaaJ = 0; fsaaJ < configuration.fsaaY; fsaaJ++) {
			// __________________________________________________
			// Getting samples for the ray:
			RaySamples cameraPixelSamples;
			{
				Vector2 sampleOffset(
					(((float)fsaaI) + 0.5f) * pixelSampleW * pixelSize,
					(((float)fsaaJ) + 0.5f) * pixelSampleH * pixelSize);
				screenSpacePosition = (offset + sampleOffset);

				LenseGetPixelSamplesRequest cameraPixelSamplesRequest;
				cameraPixelSamplesRequest.screenSpacePosition = screenSpacePosition;
				cameraPixelSamplesRequest.pixelSize = pixelSize;
				cameraPixelSamplesRequest.context = (&renderContext);
				configuration.camera->getPixelSamples(cameraPixelSamplesRequest, &cameraPixelSamples);
			}

			// Resulting pixel color:
			Color color(0.0f, 0.0f, 0.0f, 0.0f);

			// __________________________________________________
			// Actual render loop:
			int currentLayer = -1;
			while (true) {
				// ----------------------------------------------

				// If current layer is 'negative', it means that the render has not started yet, 
				// or it ended, or there are still some camera samples left to investigate:
				if (currentLayer < 0) {
					// If there are no more camera samples, we end the render process:
					if (cameraPixelSamples.sampleCount <= 0) break;

					// Else, we set the first layer up with a camera sample and decrease the counter:
					cameraPixelSamples.sampleCount--;
					currentLayer = 0;
					bounceLayers[currentLayer].setup(
						cameraPixelSamples.samples[cameraPixelSamples.sampleCount], 1.0f);
					lightRays.sampleCount = 0;
				}

				// For convenience, current bounce layer will be reffered to as 'layer':
				BounceLayer &layer = bounceLayers[currentLayer];


				// ----------------------------------------------

				// We may need to cast the ray either to investigate the direction of the layer,
				// or illuminate it. Regardless, we'll need a reference to a ray to cast (this way we decrease diverency):
				const Ray *rayToCast;
				void *restrictionObject = NULL;

				// We may need to call the shader's getReflectedColor for aome amount of photons:
				int numIlluminationPhotons = 0;

				// In case there's no geometry set, rayToCast is the layer ray:
				if (layer.geometry.object == NULL) {
					rayToCast = (&layer.layerRay);
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
				else if (layer.lightIndex < configuration.context.lights->size()) {
					// __TODO__: add light samples here...
					bool castShadows = true;
					LightVertexSampleRequest request;
					request.point = layer.geometry.hitPoint;
					request.context = (&renderContext);
					configuration.context.lights->operator[](layer.lightIndex).getVertexPhotons(
						request, &lightRays, &castShadows);
					layer.lightIndex++;

					if (castShadows) continue;
					else {
						numIlluminationPhotons = lightRays.sampleCount;
						lightRays.sampleCount = 0;
						rayToCast = NULL;
					}
				}

				// With geometry already investigated and no direct illumination photons left,
				// all we have to do is to either go on and setup the indirect sample if it's still pending:
				else if (layer.bounces.sampleCount > 0) {
					layer.bounces.sampleCount--;
					currentLayer++;
					bounceLayers[currentLayer].setup(
						layer.bounces.samples[layer.bounces.sampleCount], layer.absoluteWeight);
					lightRays.sampleCount = 0;
					continue;
				}

				// In case, there are no more indirect samples requested, all can do is to go down a layer:
				else {
					if (currentLayer > 0) {
						// Illuminate the underlying geometry:
						BounceLayer &layerBelow = bounceLayers[currentLayer - 1];
						ShaderReflectedColorRequest<SceneType::SurfaceUnit> request;
						request.object = &layerBelow.geometry.object->object;
						request.photon = Photon(Ray(layerBelow.geometry.hitPoint, -layer.layerRay.direction), layer.color);
						request.hitPoint = layerBelow.geometry.hitPoint;
						request.observerDirection = (-layerBelow.layerRay.direction);
						request.photonType = PHOTON_TYPE_INDIRECT_ILLUMINATION;
						request.significance = layer.sampleSignificance;
						request.sampleType = layer.sampleType;
						request.context = (&renderContext);
						layerBelow.color += (configuration.context.materials->operator[](
							layerBelow.geometry.object->materialId).getReflectedColor(request) * layer.sampleWeight);
					}
					else {
						// Add color to the final pixel:
						LenseGetPixelColorRequest request;
						request.screenSpacePosition = screenSpacePosition;
						request.pixelSize = pixelSize;
						request.photon = Photon(Ray(layer.geometry.hitPoint, -layer.layerRay.direction), layer.color);
						request.photonType = PHOTON_TYPE_INDIRECT_ILLUMINATION;
						request.context = (&renderContext);
						color += (configuration.camera->getPixelColor(request) * layer.sampleWeight);
					}
					currentLayer--;
					continue;
				}




				// ----------------------------------------------
				// Raycast:
				if (rayToCast != NULL) {
					// If we have a ray to cast, we should be here at some point:
					RaycastHit<SceneType::GeometryUnit> hit;
					if (configuration.context.geometry->cast(
						*rayToCast, hit, false, Octree<SceneType::GeometryUnit>::validateNotSameAsObject, restrictionObject)) {
						// If raycast hit something and it was a geometry ray,
						// we need to set the layer up:
						if (rayToCast == (&layer.layerRay)) {
							layer.geometry = hit;
							// If we are allawed to further request indirect illumination,
							// here's the time:
							if (currentLayer < maxLayer) {
								ShaderIndirectSamplesRequest<SceneType::SurfaceUnit> request;
								request.absoluteSampleWeight = layer.absoluteWeight;
								request.hitDistance = hit.hitDistance;
								request.hitPoint = hit.hitPoint;
								request.object = &hit.object->object;
								request.ray = (*rayToCast);
								request.significance = layer.sampleSignificance;
								request.sampleType = layer.sampleType;
								request.context = (&renderContext);
								configuration.context.materials->operator[](
									hit.object->materialId).requestIndirectSamples(request, &layer.bounces);
							}
							continue;
						}
						// If the ray was an illumination ray and it hit the same old object or point, we count it as a light:
						else if (
							(hit.object == layer.geometry.object) ||
							((hit.hitPoint - layer.geometry.hitPoint).sqrMagnitude() <= (8.0f * VECTOR_EPSILON))) {
							numIlluminationPhotons = 1;
						}
						// If no illumination occured whatsoever, there's no point continuing the cycle:
						else continue;
					}
					// If raycast failed and it was a geometry ray, we have to give up on the current layer:
					else if (rayToCast == (&layer.layerRay)) {
						currentLayer--;
						continue;
					}
					// If raycast failed and it was a light ray, we probably don't care about it all that much,
					// even though ity should never happen in theory:
					else continue;
				}



				// ----------------------------------------------
				// Illumination:
				const SceneType::MaterialType &layerMaterial = configuration.context.materials->operator[](layer.geometry.object->materialId);
				for (int i = 0; i < numIlluminationPhotons; i++) {
					ShaderReflectedColorRequest<SceneType::SurfaceUnit> request;
					request.object = &layer.geometry.object->object;
					request.photon = lightRays.samples[lightRays.sampleCount + i];
					request.hitPoint = layer.geometry.hitPoint;
					request.observerDirection = (-layer.layerRay.direction);
					request.photonType = PHOTON_TYPE_DIRECT_ILLUMINATION;
					request.significance = 1.0f;
					request.sampleType = 0;
					request.context = (&renderContext);
					layer.color += layerMaterial.getReflectedColor(request);
				}
			}
			result += (color * (pixelSampleW * pixelSampleH));
		}

	
	// ###################################################################
	// SETTING FINAL COLOR
	if (configuration.blendingAmount >= 1.0f)
		configuration.buffer->setBlockPixelColor(blockId, pixelId, result);
	else configuration.buffer->blendBlockPixelColor(blockId, pixelId, result, configuration.blendingAmount);
}
