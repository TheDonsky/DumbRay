#pragma once
#include "Managedhandler.cuh"
#include "Scene.cuh"
#include <mutex>


template<typename HitType>
class SceneHandler : public ManagedHandler<Scene<HitType> > {
public:
	inline SceneHandler(const Scene<HitType> &scene) : ManagedHandler<Scene<HitType> >(scene) {}
};