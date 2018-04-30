#pragma once
#include "../BlockRenderer/BlockRenderer.cuh"
#include "../../Objects/Scene/DumbScene.cuh"
#include "../../Objects/Scene/Camera/Camera.cuh"


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
		int maxBounces = 3);

	void setScene(SceneType *scene);
	SceneType* getScene()const;

	void setCamera(CameraManager *camera);
	CameraManager* getCamera()const;

	void setBoxingMode(BoxingMode mode);
	BoxingMode getBoxingMode()const;

	void setMaxBounces(int maxBounces);
	int getMaxBounces()const;

	static int maxBouncesLimit();


protected:
	bool renderBlocksCPU(const Info &info, FrameBuffer *buffer, int startBlock, int endBlock);
	bool renderBlocksGPU(const Info &info, FrameBuffer *host, FrameBuffer *device, int startBlock, int endBlock, cudaStream_t &renderStream);


private:
	volatile SceneType volatile *sceneManager;
	volatile CameraManager volatile *cameraManager;
	volatile BoxingMode boxing;
	volatile int bounceLimit;

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

			bool host(SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, BoxingMode boxingMode, float blending, int bounces);
			bool device(SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, FrameBuffer *hostFrameBuffer, BoxingMode boxingMode, int deviceId, float blending, int bounces);
			bool hasError();
		};

		__device__ __host__ void configure(const SceneConfiguration &config);
		__device__ __host__ bool setPixel(int blockId, int pixelId);

		__device__ __host__ void render();


	private:
		SceneConfiguration configuration;
		int block, pixelInBlock;
		int pixelX, pixelY;

		enum RayType {
			GEOMETRY_RAY,
			LIGHT_RAY
		};

		struct BounceLayer {
			Color color;
			Ray layerRay;
			float sampleWeight;
			float absoluteWeight;
			int lightIndex;
			RaycastHit<SceneType::GeometryUnit> geometry;
			RaySamples bounces;

			__device__ __host__ inline void setup(const SampleRay &sample, float absWeight) {
				color = Color(0.0f, 0.0f, 0.0f, 0.0f);
				layerRay = sample.ray;
				sampleWeight = sample.sampleWeight;
				absoluteWeight = sampleWeight * absWeight;
				lightIndex = 0;
				geometry.object = NULL;
				bounces.sampleCount = 0;
			}
		};
	};
};
