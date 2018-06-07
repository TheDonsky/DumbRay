#pragma once


#include"../Light.cuh"



struct SimpleSoftDirectionalLight {
	Color color;
	Vector3 dir;
	float dist;
	float soft;


	__dumb__ SimpleSoftDirectionalLight(const Color shade = Color(1.0f, 1.0f, 1.0f), const Vector3 &direction = Vector3(1, -1, 1).normalized(), float distance = 1024.0f, float softness = 0.125f);
	__dumb__ void getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const;


	inline bool fromDson(const Dson::Object &object, std::ostream *errorStream);
};



#include "SimpleSoftDirectionalLight.impl.cuh"
