#pragma once

#include"../Ray/Ray.h"
#include"../../Pure/ColorRGB/ColorRGB.cuh"
#include"../../../Objects/Components/Transform/Transform.h"




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
