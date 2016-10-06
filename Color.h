#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<fstream>
#include<iostream>

struct Color{
	/** -------------------------------------------------------------------------- **/
	/** Parameters: **/
	float r, g, b, a;





	/** -------------------------------------------------------------------------- **/
	/** Construction: **/
	__device__ __host__ inline Color();
	__device__ __host__ inline Color(float R, float G, float B, float A = 1);
	__device__ __host__ inline Color(const Color &c);
	__device__ __host__ inline Color& operator()(float R, float G, float B, float A = 1);
};





/** -------------------------------------------------------------------------- **/
/** Stream operators: **/
std::istream& operator>>(std::istream &stream, Color &r);
std::ostream& operator<<(std::ostream &stream, const Color &r);


#include"Color.impl.h"