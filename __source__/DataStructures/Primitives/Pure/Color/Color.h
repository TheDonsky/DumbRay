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





	/** -------------------------------------------------------------------------- **/
	/** Operators: **/
	/* 
	Note: these are vector-like operations and should not be treated as
	valid things for color specific calculations...
	*/
	__device__ __host__ inline Color operator+()const;
	__device__ __host__ inline Color operator+(const Color &c)const;
	__device__ __host__ inline Color& operator+=(const Color &c);

	__device__ __host__ inline Color operator-()const;
	__device__ __host__ inline Color operator-(const Color &c)const;
	__device__ __host__ inline Color& operator-=(const Color &c);

	__device__ __host__ inline Color operator*(const Color &c)const;
	__device__ __host__ inline Color operator*(float f)const;
	__device__ __host__ inline Color& operator*=(const Color &c);
	__device__ __host__ inline Color& operator*=(float f);

	__device__ __host__ inline Color operator/(const Color &c)const;
	__device__ __host__ inline Color operator/(float f)const;
	__device__ __host__ inline Color& operator/=(const Color &c);
	__device__ __host__ inline Color& operator/=(float f);
};





/** -------------------------------------------------------------------------- **/
/** Stream operators: **/
inline static std::istream& operator>>(std::istream &stream, Color &r);
inline static std::ostream& operator<<(std::ostream &stream, const Color &r);


#include"Color.impl.h"
