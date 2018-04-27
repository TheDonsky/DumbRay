#pragma once
#include "../BlockRenderer/BlockRenderer.cuh"
#include "../../Objects/Scene/DumbScene.cuh"
#include "../../Objects/Scene/Camera/Camera.cuh"


class DumbRenderer : public BlockRenderer {
public:
	typedef TriScene SceneType;
	typedef ReferenceManager<Camera> CameraManager;

	DumbRenderer(
		const ThreadConfiguration &configuration = ThreadConfiguration(ThreadConfiguration::ALL, 2),
		const BlockConfiguration &blockSettings = BlockConfiguration(),
		FrameBufferManager *buffer = NULL,
		SceneType *scene = NULL,
		CameraManager *camera = NULL);

	void setScene(SceneType *scene);
	SceneType* getScene()const;

	void setCamera(CameraManager *camera);
	CameraManager* getCamera()const;


protected:
	bool renderBlocksCPU(const Info &info, FrameBuffer *buffer, int startBlock, int endBlock);
	bool renderBlocksGPU(const Info &info, FrameBuffer *host, FrameBuffer *device, int startBlock, int endBlock, cudaStream_t &renderStream);


private:
	SceneType *sceneManager;
	CameraManager *cameraManager;

public:
	class PixelRenderProcess {
	public:
		struct SceneConfiguration {
			SceneType::Context context;
			Camera *camera;
			FrameBuffer *buffer;

			bool host(SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer);
			bool device(SceneType *scene, CameraManager *cameraManager, FrameBuffer *frameBuffer, int deviceId);
			bool hasError();
		};

		__device__ __host__ void configure(const SceneConfiguration &config);
		__device__ __host__ bool setPixel(int blockId, int pixelId);
		__device__ __host__ void reset();

		__device__ __host__ void render();


	private:
		SceneConfiguration configuration;
		int block, pixelInBlock;
		int pixelX, pixelY;
	};
};
