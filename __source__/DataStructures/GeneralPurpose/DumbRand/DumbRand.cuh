#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<stdint.h>
#include<stdlib.h>
#include<mutex>

/* 
NOTE TO ANYONE WHO HAPPENS TO BE USING THIS:
The algorithm is XORWOW. Aka whatever the default thing for official curand is.
However, all the functions included here are __divice__ __host__ instead of just __device__, 
renderering the target platform somewhat irrelevant when coding...
Copying already existing rng-s may or may not be stupid, but this time, I'm somewhat ok with that.
*/

class DumbRand {
public:
	typedef uint32_t UnsignedInt;
	typedef int32_t SignedInt;

	// Default constructor (does absolutely nothing)
	__device__ __host__ inline DumbRand();

	// Constructor that seeds all five seed values
	__device__ __host__ inline DumbRand(
		UnsignedInt seedA, UnsignedInt seedB = 1, UnsignedInt seedC = 1, UnsignedInt seedD = 1, UnsignedInt seedE = 0);

	// Seeds with all five seed values
	__device__ __host__ inline void seed(
		UnsignedInt seedA, UnsignedInt seedB = 1, UnsignedInt seedC = 1, UnsignedInt seedD = 1, UnsignedInt seedE = 0);

	// Seeds with rand(:
	inline void seed();

	// Generates random unit
	__device__ __host__ inline UnsignedInt get();

	// Generates signed integer
	__device__ __host__ inline SignedInt getInt();

	// Unsigned range between minimum (inclusive) and maximum (exclusive) values
	__device__ __host__ inline UnsignedInt rangeUnsigned(UnsignedInt minimum, UnsignedInt maximum);

	// Signed range between minimum (inclusive) and maximum (exclusive) values
	__device__ __host__ inline UnsignedInt rangeSigned(SignedInt minimum, SignedInt maximum);

	// Random float between 0 (inclusive) and 1 (inclusive):
	__device__ __host__ inline float getFloat();

	// Random float between minimum (inclusive) and maximum (includive) values
	__device__ __host__ inline float range(float minimum, float maximum);



private:
	static std::mutex lock;
	UnsignedInt a, b, c, d, e;
};



#include"DumbRand.impl.cuh"
