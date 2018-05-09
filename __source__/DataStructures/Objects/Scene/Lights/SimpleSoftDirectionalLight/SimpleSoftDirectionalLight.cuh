#pragma once


#include"../Light.cuh"



struct SimpleSoftDirectionalLight {
	Color color;
	Vector3 dir;
	float dist;
	float soft;


	__dumb__ SimpleSoftDirectionalLight(const Color shade, const Vector3 &direction, float distance, float softness = 32.0f);
	__dumb__ void getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const;
};



#include "SimpleSoftDirectionalLight.impl.cuh"
