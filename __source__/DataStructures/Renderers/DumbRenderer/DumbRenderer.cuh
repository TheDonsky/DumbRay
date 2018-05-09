#pragma once
#include "../BlockRenderer/BlockRenderer.cuh"
#include "../../Objects/Scene/DumbScene.cuh"
#include "../../Objects/Scene/Camera/Camera.cuh"
#include "../../GeneralPurpose/DumbRand/DumbRand.cuh"


class DumbRenderer : public BlockRenderer {
public:
	typedef TriScene SceneType;
	typedef ReferenceManager<Camera> CameraManager;
	enum BoxingMode {
		BOXING_MODE_HEIGHT_BASED = 0,
		BOXING_MODE_WIDTH_BASED = 1,
		BOXING_MODE_MIN_BASED = 2,
		BOXING_MODE_MAX_BASED = 3
	};

	DumbRenderer(
		const ThreadConfiguration &configuration = ThreadConfiguration(ThreadConfiguration::ALL, 2),
		const BlockConfiguration &blockSettings = BlockConfiguration(),
		FrameBufferManager *buffer = NULL,
		SceneType *scene = NULL,
		CameraManager *camera = NULL,
		BoxingMode boxingMode = BOXING_MODE_HEIGHT_BASED,
		int maxBounces = 2,
		int samplesPerPixelX = 1,
		int samplesPerPixelY = 1);

	void setScene(SceneType *scene);
	SceneType* getScene()const;

	void setCamera(CameraManager *camera);
	CameraManager* getCamera()const;

	void setBoxingMode(BoxingMode mode);
	BoxingMode getBoxingMode()const;

	void setMaxBounces(int maxBounces);
	int getMaxBounces()const;

	static int maxBouncesLimit();

	void setSamplesPerPixelX(int value);
	void setSamplesPerPixelY(int value);
	void setSamplesPerPixel(int x, int y);
	int getSamplesPerPixelX()const;
	int getSamplesPerPixelY()const;



protected:
	bool renderBlocksCPU(const Info &info, FrameBuffer *buffer, int startBlock, int endBlock);
	bool renderBlocksGPU(const Info &info, FrameBuffer *host, FrameBuffer *device, int startBlock, int endBlock, cudaStream_t &renderStream);


private:
	volatile SceneType *sceneManager;
	volatile CameraManager *cameraManager;
	volatile BoxingMode boxing;
	volatile int bounceLimit;
	volatile int fsaaX, fsaaY;

public:
	class PixelRenderProcess {
	public:
		struct SceneConfiguration {
			SceneType::Context context;
			Camera *camera;
			FrameBuffer *buffer;
			BoxingMode boxing;
			int width, height;
			float blendingAmount;
			int maxBounces;
			int fsaaX, fsaaY;

			bool host(
				SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, 
				BoxingMode boxingMode, float blending, int bounces, int samplesX, int samplesY);
			bool device(
				SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, FrameBuffer *hostFrameBuffer, 
				BoxingMode boxingMode, int deviceId, float blending, int bounces, int samplesX, int samplesY);
			bool hasError();
		};

		__device__ __host__ void configure(const SceneConfiguration &config);
		__device__ __host__ void setContext(const RenderContext &context, int entropyOffset);
		__device__ __host__ void renderPixel(int blockId, int pixelId);


	private:
		SceneConfiguration configuration;
		RenderContext renderContext;

		enum RayType {
			GEOMETRY_RAY,
			LIGHT_RAY
		};

		struct BounceLayer {
			Color color;
			Ray layerRay;
			int lightIndex;
			float sampleWeight;
			float absoluteWeight;
			RaycastHit<SceneType::GeometryUnit> geometry;
			RaySamples bounces;

			__device__ __host__ inline void setup(const SampleRay &sample, float absWeight) {
				color = Color(0.0f, 0.0f, 0.0f, 0.0f);
				layerRay = sample.ray;
				sampleWeight = sample.sampleWeight;
				absoluteWeight = sampleWeight * absWeight;
				geometry.object = NULL;
				lightIndex = 0;
				bounces.sampleCount = 0;
			}
		};
	};
};