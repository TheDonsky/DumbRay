#pragma once
#include"Light.cuh"



struct SimpleDirectionalLight {
	Photon photon;

	__dumb__ SimpleDirectionalLight(Photon photon);
	__dumb__ Photon getPhoton(const Vertex &targetPoint, bool *noShadows)const;
	__dumb__ ColorRGB ambient(const Vertex &targetPoint)const;
};



#include"SimpleDirectionalLight.impl.cuh"
