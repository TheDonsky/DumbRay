#pragma once
#include "../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include "../Renderer/Renderer.cuh"
#include "../../Objects/Scene/SceneHandler/SceneHandler.cuh"
#include "../../../Namespaces/Device/Device.cuh"
#include "../../Screen/FrameBuffer/FrameBuffer.cuh"


#define BACKWARD_RENDERER_MAX_BOUNCES 64


template<typename HitType>
class BackwardRenderer : public Renderer{
public:
	enum BoxingType {
		VERTICAL,	// Causes the horizontal viewing angle to be dependent on vertical resolution
		HORIZONTAL,	// Causes the vertical viewing angle to be dependent on horizontal resolution
		MINIMAL,	// Viewing angle will be dependent on the smaller resolution
		MAXIMAL,	// Viewing angle will be dependent on the bigger resolution
		NONE		// No scaling, whatsoever... I don't know... May be useful for something down the line
	};
	struct Configuration {
		SceneHandler<HitType> *scene;
		int maxBounces;
		int multisampling;
		int segmentWidth, segmentHeight;
		BoxingType boxingType;
		int cameraId;
		int deviceOverstauration;
		int devicePixelsPerThread;
		int hostOversaturation;

		// Configuration. (Note: BackwardRenderer is not responsible for the consequences caused by incorrect values from here)
		inline Configuration(
			SceneHandler<HitType> &s,					// Scene handle;
			int maxBounce = 4,							// Maximum amount of theorhetical bounces, before thre renderer gives up on the photon (can not be more than BACKWARD_RENDERER_MAX_BOUNCES).
			int samples = 1,							// Multisampling (each pixel becomes average of (samples by samples) matrix)
			int blockWidth = 16, int blockHeight = 16,	// Block size (consider, that blockWidth * blockHeight has to be less or equal to 512 in order for a GPU thread to stand a chance to execute);
			BoxingType boxing = VERTICAL,				// Image boxing (messes with lense behaviour for non-quadratic render targets)
			int camera = 0,								// Selected camera index
			int devOversaturation = 8,					// Defines, how okversaturated a single streaming multiprocessor will be on average (more will likely give better performance, but keep in mind that this one eats up a lot of VRAM)
			int pixelPerDeviceThread = 32,				// Defines, how many pixels each GPU thread will be tasked with.
			int cpuOversaturation = 1					// Defines, how many blocks a CPU core will take at once (Higher number will likely boost performance, when there's no GPU).
		);
	};
	
	inline BackwardRenderer(
		const Configuration &configuration, 
		const ThreadConfiguration &threads = ThreadConfiguration());
	inline virtual ~BackwardRenderer();

	inline bool selectCamera(int index);
	inline void setBoxingType(BoxingType boxing);
	inline void setFrameBuffer(FrameBufferManager &manager);


protected:
	inline virtual bool setupSharedData(const Info &info, void *&sharedData);
	inline virtual bool setupData(const Info &info, void *&data);
	inline virtual bool prepareIteration();
	inline virtual void iterateCPU(const Info &info);
	inline virtual void iterateGPU(const Info &info);
	inline virtual bool completeIteration();
	inline virtual bool clearData(const Info &info, void *&data);
	inline virtual bool clearSharedData(const Info &info, void *&sharedData);

public:
	struct PixelRenderProcess {
		struct BounceObject {
			RaycastHit<Renderable<HitType> > hit;
			Photon bounce;
			ColorRGB color;
			PhotonPack bounces;
		};
		BounceObject bounces[BACKWARD_RENDERER_MAX_BOUNCES + 1];
		PhotonPack lightIllumination;
		Vector2 screenPoint;
		int posX, posY;
		bool stateActive;
		Color color;
		int bounceHeight;
		int lightId;


		struct Settings {
			int maxBounces;
			int multisampling;
			BoxingType boxing;
			float blendAmount;

			__dumb__ Settings(
				int bounce = 0, int samples = 1, 
				BoxingType box = VERTICAL, float blend = 1.0f);
		};
		Settings settings;

		struct Context {
			const Scene<HitType> *scene;
			const Camera *camera;
			FrameBuffer *frame;
			int width, height;

			__dumb__ Context(
				const Scene<HitType> *s = NULL, 
				const Camera *c = NULL, FrameBuffer *f = NULL);
		};
		Context context;

		struct Domain {
			int startBlock, endBlock;
			int blockWidth, blockHeight;
			int pixelIndex;

			__dumb__ Domain(
				int startB = -1, int endB = -1,
				int bWidth = 1, int bHeight = 1,
				int pixelId = 1024);
		};
		Domain domain;

		struct State {
			int block;
			int sampleX, sampleY;
			float sampleDelta;
			float sampleMass;
		};
		State state;
		
		__dumb__ bool countBase();
		__dumb__ bool moveState();
		__dumb__ void resetState(); // This should be called manually;
		__dumb__ void initPixel(float x, float y);
		__dumb__ bool renderPixel(int &raycastBudget);
		__dumb__ bool render(int &raycastBudget); // This should be called manually;
	};


private:
	struct DeviceThreadData {
		PixelRenderProcess *pixels;
		int blockCount;
		int raycastBudget;
		bool *renderEnded;
		cudaStream_t stream;
		BackwardRenderer *renderer;

		inline DeviceThreadData(BackwardRenderer *owner);
		inline ~DeviceThreadData();

		inline int pixelCount()const;
		inline int blockSize()const;
	};

	struct SegmentBank {
		int blockCount;
		int counter;
		std::mutex lock;

		inline void reset(int totalBlocks);
		inline bool get(int count, int &start, int &end);
	};

private:
	Configuration config;
	SegmentBank segmentBank;
	FrameBufferManager *frameBuffer;
};









#include "BackwardRenderer.impl.cuh"


