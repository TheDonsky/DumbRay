#pragma once
#include"ShadedOctree.cuh"
#include"Light.cuh"
#include"Camera.cuh"
#include"Handler.cuh"
#include"Matrix.h"


template<typename HitType> struct Scene;
template<typename HitType>
class TypeTools<Scene<HitType> > {
public:
	typedef Scene<HitType> SceneType;
	typedef ShadedOctree<HitType> PartType0;
	typedef Stacktor<Light> PartType1;
	typedef Stacktor<Camera> PartType2;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(SceneType);
};

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
