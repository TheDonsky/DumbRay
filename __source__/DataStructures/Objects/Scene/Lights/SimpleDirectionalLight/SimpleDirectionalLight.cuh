#pragma once
#include"../Light.cuh"



struct SimpleDirectionalLight {
	Photon photon;

	__dumb__ SimpleDirectionalLight(Photon photon);
	__dumb__ void getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const;
};



#include"SimpleDirectionalLight.impl.cuh"
