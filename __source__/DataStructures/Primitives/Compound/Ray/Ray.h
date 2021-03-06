#pragma once

#include"../../Pure/Vector3/Vector3.h"

//#define INFINITY 999999999.9f

struct Ray{
	/** -------------------------------------------------------------------------- **/
	/** Parameters: **/
	Vertex origin;
	Vector3 direction;





	/** -------------------------------------------------------------------------- **/
	/** Construction: **/
	__device__ __host__ inline Ray();
	__device__ __host__ inline Ray(const Vector3 &org, const Vector3 &dir);
	__device__ __host__ inline Ray(const Ray &r);
	__device__ __host__ inline Ray& operator()(const Vector3 &org, const Vector3 &dir);
};





/** -------------------------------------------------------------------------- **/
/** Stream operators: **/
inline static std::istream& operator>>(std::istream &stream, Ray &r);
inline static std::ostream& operator<<(std::ostream &stream, const Ray &r);


#include"Ray.impl.h"
