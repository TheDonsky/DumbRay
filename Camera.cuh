#pragma once
#include"Transform.h"
#include"Photon.cuh"
#include"Vector2.h"
#include"Lense.cuh"




struct Camera {
	Transform transform;
	Lense lense;

	__dumb__ Photon getPhoton(const Vector2 &screenSpacePosition)const;
};


SPECIALISE_TYPE_TOOLS_FOR(Camera);



#include"Camera.impl.cuh"

