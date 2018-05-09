#pragma once
#include"../Light.cuh"



struct SimpleDirectionalLight {
	Color color;
	Vector3 dir;
	float dist;


	__dumb__ SimpleDirectionalLight(const Color shade, const Vector3 &direction, float distance);
	__dumb__ void getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const;
};



#include"SimpleDirectionalLight.impl.cuh"
