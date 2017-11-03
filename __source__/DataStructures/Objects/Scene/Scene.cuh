#pragma once
#include"Raycasters/ShadedOctree/ShadedOctree.cuh"
#include"Lights/Light.cuh"
#include"Camera/Camera.cuh"
#include"../../GeneralPurpose/Handler/Handler.cuh"
#include"../../GeneralPurpose/Matrix/Matrix.h"

//#define SCENE_USE_GENERIC_RAYCASTER

template<typename HitType> struct Scene;
#ifdef SCENE_USE_GENERIC_RAYCASTER
TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE(Scene, Raycaster<Shaded<HitType> >, Stacktor<Light>, Stacktor<Camera>, typename HitType);
#else
TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE(Scene, ShadedOctree<HitType>, Stacktor<Light>, Stacktor<Camera>, typename HitType);
#endif

template<typename HitType>
struct Scene {
#ifdef SCENE_USE_GENERIC_RAYCASTER
	Raycaster<Shaded<HitType> > geometry;
#else
	ShadedOctree<HitType> geometry;
#endif
	Stacktor<Light> lights;
	Stacktor<Camera> cameras;

	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	DEFINE_CUDA_LOAD_INTERFACE_FOR(Scene);
	TYPE_TOOLS_ADD_COMPONENT_GETTERS_3(Scene, geometry, lights, cameras);
};






#include"Scene.impl.cuh"