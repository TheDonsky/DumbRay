#pragma once

#include"Ray.h"
#include"ColorRGB.cuh"




struct Photon{
	Ray ray;
	ColorRGB color;

	__dumb__ Photon();
	__dumb__ Photon(const Ray &r, const ColorRGB &c);



	__device__ __host__ inline Photon operator>>(const Transform &trans)const;
	__device__ __host__ inline Photon& operator>>=(const Transform &trans);
	__device__ __host__ inline Photon operator<<(const Transform &trans)const;
	__device__ __host__ inline Photon& operator<<=(const Transform &trans);



	__dumb__ float energy()const;
	__dumb__ float dead()const;


	__dumb__ static Photon zero();
};





#include"Photon.impl.cuh"
