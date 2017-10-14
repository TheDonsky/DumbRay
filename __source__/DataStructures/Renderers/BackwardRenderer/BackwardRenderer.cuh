#pragma once
#include "Renderer.cuh"
#include "SceneHandler.cuh"


#define BACKWARD_RENDERER_MAX_BOUNCES 128;



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
	struct PixelState {
		struct Frame {
			Photon photon;
			typename ShadedOctree<HitType>::RaycastHit hit;
			ShaderBounce bounce;
		};
		PixelState stack[BACKWARD_RENDERER_MAX_BOUNCES];
		PixelState *ptr;
		PixelState *end;
		__dumb__ void reset();
		__dumb__ void cast(Photon photon);
	};

private:
	SceneHandler<HitType> *data;
	int selectedCamera;
};









#include "BackwardRenderer.impl.cuh"
