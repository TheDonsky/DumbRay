#pragma once

#include"Vector3.h"
#include"Color.h"


struct ColorRGB {
	/** -------------------------------------------------------------------------- **/
	/** Parameters: **/
	float r, g, b;





	/** -------------------------------------------------------------------------- **/
	/** Construction: **/
	__device__ __host__ inline ColorRGB();
	__device__ __host__ inline ColorRGB(float R, float G, float B);
	__device__ __host__ inline ColorRGB(const ColorRGB &c);
	__device__ __host__ inline ColorRGB& operator()(float R, float G, float B);





	/** -------------------------------------------------------------------------- **/
	/** Casts: **/
	__dumb__ operator Color()const;
	__dumb__ operator Vector3()const;
	__dumb__ ColorRGB(const Color &c);
	__dumb__ ColorRGB(const Vector3 &v);





	/** -------------------------------------------------------------------------- **/
	/** Operators: **/

	__dumb__ ColorRGB operator+()const;
	__dumb__ ColorRGB operator+(const ColorRGB &c)const;
	__dumb__ ColorRGB& operator+=(const ColorRGB &c);

	__dumb__ ColorRGB operator-()const;
	__dumb__ ColorRGB operator-(const ColorRGB &c)const;
	__dumb__ ColorRGB& operator-=(const ColorRGB &c);

	__dumb__ ColorRGB operator*(const ColorRGB &c)const;
	__dumb__ ColorRGB operator*(float f)const;
	__dumb__ ColorRGB& operator*=(const ColorRGB &c);
	__dumb__ ColorRGB& operator*=(float f);

	__dumb__ ColorRGB operator/(const ColorRGB &c)const;
	__dumb__ ColorRGB operator/(float f)const;
	__dumb__ ColorRGB& operator/=(const ColorRGB &c);
	__dumb__ ColorRGB& operator/=(float f);
};





/** -------------------------------------------------------------------------- **/
/** Stream operators: **/
inline static std::istream& operator >> (std::istream &stream, ColorRGB &r);
inline static std::ostream& operator<<(std::ostream &stream, const ColorRGB &r);



#include"ColorRGB.impl.cuh"

