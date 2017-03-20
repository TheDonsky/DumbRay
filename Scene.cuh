#pragma once
#include"ShadedOctree.cuh"
#include"Light.cuh"
#include"Camera.cuh"
#include"Handler.cuh"
#include"Matrix.h"


template<typename HitType> struct Scene;
TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE(Scene, ShadedOctree<HitType>, Stacktor<Light>, Stacktor<Camera>, typename HitType);

template<typename HitType>
struct Scene {
	ShadedOctree<HitType> geometry;
	Stacktor<Light> lights;
	Stacktor<Camera> cameras;

	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	DEFINE_CUDA_LOAD_INTERFACE_FOR(Scene);
};






#include"Scene.impl.cuh"
