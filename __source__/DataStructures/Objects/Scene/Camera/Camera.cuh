#pragma once
#include"../../Components/Transform/Transform.h"
#include"../../../Primitives/Compound/Photon/Photon.cuh"
#include"../../../Primitives/Pure/Vector2/Vector2.h"
#include"../../Components/Lenses/Lense.cuh"


class Camera;
TYPE_TOOLS_REDEFINE_2_PART(Camera, Transform, Lense);


class Camera {
public:
	Transform transform;
	Lense lense;

	__dumb__ void getPhoton(const Vector2 &screenSpacePosition, PhotonPack &result)const;

	// For upload:
	DEFINE_CUDA_LOAD_INTERFACE_FOR(Camera);
private:
	TYPE_TOOLS_ADD_COMPONENT_GETTERS_2(Camera, transform, lense);
};





#include"Camera.impl.cuh"

