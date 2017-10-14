#pragma once
#include "../../../GeneralPurpose/ManagedHandler/ManagedHandler.cuh"
#include "../Scene.cuh"
#include <mutex>


template<typename HitType>
class SceneHandler : public ManagedHandler<Scene<HitType> > {
public:
	inline SceneHandler(const Scene<HitType> &scene) : ManagedHandler<Scene<HitType> >(scene) {}
};