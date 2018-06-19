#pragma once
#include"../Light.cuh"
#include "../../../../DumbRenderContext/DumbRenderContext.cuh"





struct SphericalLightEmitter {
	Color col;
	Vector3 pos;
	float rad;

	__dumb__ SphericalLightEmitter(const Color shade = Color(1.0f, 1.0f, 1.0f), const Vector3 &position = Vector3(0.0f, 0.0f, 0.0f), float radius = 1.0f);
	__dumb__ void getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const;


	inline bool fromDson(const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context);
};



#include "SphericalLightEmitter.impl.cuh"