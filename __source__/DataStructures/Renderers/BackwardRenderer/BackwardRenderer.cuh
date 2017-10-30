#pragma once
#include "../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include "../Renderer/Renderer.cuh"
#include "../../Objects/Scene/SceneHandler/SceneHandler.cuh"


#define BACKWARD_RENDERER_MAX_BOUNCES 64



template<typename HitType>
class BackwardRenderer : public Renderer{
public:
	inline BackwardRenderer(const ThreadConfiguration &threads, SceneHandler<HitType> &scene);
	inline virtual ~BackwardRenderer();

	inline bool selectCamera(int index);

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
		enum BoxingType {
			VERTICAL,	// Causes the horizontal viewing angle to be dependent on vertical resolution
			HORIZONTAL,	// Causes the vertical viewing angle to be dependent on horizontal resolution
			MINIMAL,	// Viewing angle will be dependent on the smaller resolution
			MAXIMAL,	// Viewing angle will be dependent on the bigger resolution
			NONE		// No scaling, whatsoever... I don't know... May be useful for something down the line
		};
		struct BounceObject {
			RaycastHit<HitType> hit;
			Photon bounce;
			ColorRGB color;
			PhotonPack bounces;
		};
		Stacktor<BounceObject, BACKWARD_RENDERER_MAX_BOUNCES + 1> bounces;
		PhotonPack lightIllumination;
		Color pixelColor;
		const Scene<HitType> *scene;
		int posX, posY;
		int maxBounces;

		__dumb__ void init(const Scene<HitType> &scn, const Camera &cmr, int x, int y, int w, int h, BoxingType boxing, int maxBnc);
		__dumb__ bool render(int &raycastBudget, int &lightCallBudget);
	};


private:
	struct DeviceThreadData {
		PixelRenderProcess *pixels;
		int blockCount;
		int raycastBudget;
		int lightCallBudget;
		bool *renderEnded;
		cudaStream_t stream;
		int blockWidth, blockHeight;

		inline DeviceThreadData(int blkWidth, int blkHeight);
		inline ~DeviceThreadData();

		inline int pixelCount()const;
	};
	
	struct HostThreadData {
		PixelRenderProcess *pixels;
		int blockWidth, blockHeight;

		inline HostThreadData(int blkWidth, int blkHeight);
		inline ~HostThreadData();

		inline int pixelCount()const;
	};

	struct SegmentBank {
		int blockCount;
		int counter;
		std::mutex lock;

		inline void reset(int totalBlocks);
		inline bool get(int count, int &start, int &end);
	}

private:
	SceneHandler<HitType> *sceneData;
	SegmentBank segmentBank;
	int selectedCamera;
};









#include "BackwardRenderer.impl.cuh"


