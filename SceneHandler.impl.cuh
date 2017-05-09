#include "SceneHandler.cuh"



template <typename HitType>
/*
Constructor.
*/
inline SceneHandler<HitType>::SceneHandler(const Scene<HitType> &s) {
	scene = (&s);
	int deviceCount;
	if (cudaGetDeviceCount(&deviceCount) == cudaSuccess)
		for (int i = 0; i < deviceCount; i++)
			deviceScenes.push(NULL);
}

template <typename HitType>
/*
Destructor.
*/
inline SceneHandler<HitType>::~SceneHandler() {
	cleanEveryGPU();
}

template <typename HitType>
/*
Sets the GPU context.
*/
inline bool SceneHandler<HitType>::selectGPU(int index)const {
	if (index < 0 || index >= deviceScenes.size()) return false;
	else return (cudaSetDevice(index) == cudaSuccess);
}

template <typename HitType>
/*
Uploads/updates the scene to the given GPU and selects it's context.
*/
inline bool SceneHandler<HitType>::uploadToGPU(int index, bool overrideExisting) {
	std::lock_guard<std::mutex> guard(lock);
	if (selectGPU(index)) {
		if (deviceScenes[index] != NULL) {
			if (overrideExisting) {
				if (!Scene<HitType>::dispose(deviceScenes[index])) return false;
				else return scene->uploadAt(deviceScenes[index]);
			}
			else return true;
		}
		else {
			deviceScenes[index] = scene->upload();
			return (deviceScenes[index] != NULL);
		}
	}
	else return false;
}

template <typename HitType>
/*
Deallocates the GPU instance (leaves context set).
*/
inline bool SceneHandler<HitType>::cleanGPU(int index) {
	std::lock_guard<std::mutex> guard(lock);
	if (selectGPU(index)) {
		if (deviceScenes[index] != NULL) {
			if (!scene->dispose(deviceScenes[index])) return false;
			else if (cudaFree(deviceScenes[index]) != cudaSuccess) return false;
			else {
				deviceScenes[index] = NULL;
				return true;
			}
		}
		else return true;
	}
	else return false;
}

template <typename HitType>
/*
Uploads/updates the scene on every available GPU (which context will stay selected, is undefined).
*/
inline void SceneHandler<HitType>::uploadToEveryGPU(bool overrideExisting) {
	for (int i = 0; i < deviceScenes.size(); i++) uploadToGPU(i, overrideExisting);
}

template <typename HitType>
/*
Deallocates every GPU instance (which context will stay selected, is undefined).
*/
inline void SceneHandler<HitType>::cleanEveryGPU() {
	for (int i = 0; i < deviceScenes.size(); i++) cleanGPU(i);
}

template <typename HitType>
/*
Returns detected GPU count.
*/
inline int SceneHandler<HitType>::gpuCount()const {
	return (deviceScenes.size());
}

template <typename HitType>
/*
Returns CPU handle.
*/
inline const Scene<HitType>* SceneHandler<HitType>::getHandleCPU()const {
	return scene;
}

template <typename HitType>
/*
Returns GPU handle (no context selection here...).
*/
inline const Scene<HitType>* SceneHandler<HitType>::getHandleGPU(int index)const {
	return deviceScenes[index];
}


