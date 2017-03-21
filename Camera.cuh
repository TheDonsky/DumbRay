#pragma once
#include"Transform.h"
#include"Photon.cuh"
#include"Vector2.h"
#include"Lense.cuh"


class Camera;
TYPE_TOOLS_REDEFINE_2_PART(Camera, Transform, Lense);


class Camera {
public:
	Transform transform;
	Lense lense;

	__dumb__ Photon getPhoton(const Vector2 &screenSpacePosition)const;

	// For upload:
	DEFINE_CUDA_LOAD_INTERFACE_FOR(Camera);
private:
	TYPE_TOOLS_ADD_COMPONENT_GETTERS_2(Camera, transform, lense);
};





#include"Camera.impl.cuh"

