#include "BackwardRenderer.cuh"


namespace BackwardRendererPrivate {

}


template<typename HitType>
inline BackwardRenderer<HitType>::BackwardRenderer(const ThreadConfiguration &threads, SceneHandler<HitType> &scene) : Renderer(threads){
	data = (&scene);
	selectedCamera = 0;
}
template<typename HitType>
inline BackwardRenderer<HitType>::~BackwardRenderer() {
	killRenderThreads();
}


template<typename HitType>
inline bool BackwardRenderer<HitType>::setImageSize(int width, int height) {
	if (width < 0 || height < 0) return false;
	resetIterations();
	// __TODO__
	return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::selectCamera(int index) {
	if (index >= 0 && index < data->getHandleCPU()->cameras.size()) {
		selectedCamera = index;
		resetIterations();
		return true;
	}
	else return false;
}

template<typename HitType>
inline bool BackwardRenderer<HitType>::setupSharedData(const Info &info, void *&sharedData) {
	if (info.isGPU()) return data->uploadToGPU(info.device, false);
	else return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::setupData(const Info &info, void *&data) {
	if (info.isGPU()) {
		if (data->getHandleGPU(info.device) == NULL) return false;
		if ((!info.manageSharedData) && (!data->selectGPU(info.device))) return false;
		// __TODO__
		return true;
	}
	else {
		if (data->getHandleCPU() == NULL) return false;
		// __TODO__
	}
	return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::prepareIteration() {
	// __TODO__
	return true;
}
template<typename HitType>
inline void BackwardRenderer<HitType>::iterateCPU(const Info &info) {
	// __TODO__
}
template<typename HitType>
inline void BackwardRenderer<HitType>::iterateGPU(const Info &info) {
	// __TODO__
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::completeIteration() {
	// __TODO__
	return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::clearData(const Info &info, void *&data) {
	// __TODO__
	return true;
}
template<typename HitType>
inline bool BackwardRenderer<HitType>::clearSharedData(const Info &info, void *&sharedData) {
	// __TODO__
	return true;
}
