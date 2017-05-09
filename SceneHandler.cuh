#pragma once
#include "Scene.cuh"
#include <mutex>



template <typename HitType>
class SceneHandler {
public:
	/*
	Constructor.
	*/
	inline SceneHandler(const Scene<HitType> &s);
	
	/*
	Destructor.
	*/
	inline ~SceneHandler();
	
	/*
	Sets the GPU context.
	*/
	inline bool selectGPU(int index)const;

	/*
	Uploads/updates the scene to the given GPU and selects it's context.
	*/
	inline bool uploadToGPU(int index, bool overrideExisting = false);

	/*
	Deallocates the GPU instance (leaves context set).
	*/
	inline bool cleanGPU(int index);

	/*
	Uploads/updates the scene on every available GPU (which context will stay selected, is undefined).
	*/
	inline void uploadToEveryGPU(bool overrideExisting = false);

	/*
	Deallocates every GPU instance (which context will stay selected, is undefined).
	*/
	inline void cleanEveryGPU();

	/*
	Returns detected GPU count.
	*/
	inline int gpuCount()const;

	/*
	Returns CPU handle.
	*/
	inline const Scene<HitType>* getHandleCPU()const;

	/*
	Returns GPU handle (no context selection here...).
	*/
	inline const Scene<HitType>* getHandleGPU(int index)const;





private:
	std::mutex lock;
	const Scene<HitType> *scene;
	Stacktor<Scene<HitType>*> deviceScenes;

	// WE RESTRICT COPY-CONSTRUCTION FOR THIS ONE...
	inline SceneHandler(const SceneHandler &other){}
	inline SceneHandler &operator=(const SceneHandler &other) { return (*this); }
};




#include "SceneHandler.impl.cuh"

