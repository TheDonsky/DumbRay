#pragma once

#include"Ray.h"
#include"ColorRGB.cuh"




struct Photon{
	Ray ray;
	ColorRGB color;

	__dumb__ Photon();
	__dumb__ Photon(const Ray &r, const ColorRGB &c);
};





#include"Photon.impl.cuh"
