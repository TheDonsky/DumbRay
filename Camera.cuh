#pragma once
#include"Transform.h"
#include"Photon.cuh"
#include"Vector2.h"
#include"Lense.cuh"


struct Camera;
template<>
class TypeTools<Camera> {
	typedef Camera MasterType;
	typedef Transform PartType0;
	typedef Lense PartType1;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType);
};


struct Camera {
	Transform transform;
	Lense lense;

	__dumb__ Photon getPhoton(const Vector2 &screenSpacePosition)const;

	// For upload:
	__dumb__ Transform &component0();
	__dumb__ const Transform &component0()const;
	__dumb__ Lense &component1();
	__dumb__ const Lense &component1()const;
	DEFINE_CUDA_LOAD_INTERFACE_FOR(Camera);
};





#include"Camera.impl.cuh"

