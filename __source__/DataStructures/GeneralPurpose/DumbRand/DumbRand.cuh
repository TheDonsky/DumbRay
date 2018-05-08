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
	// Default constructor (does absolutely nothing)
	__device__ __host__ inline DumbRand();

	// Constructor that seeds all five seed values
	__device__ __host__ inline DumbRand(
		unsigned int seedA, unsigned int seedB = 1, unsigned int seedC = 1, unsigned int seedD = 1, unsigned int seedE = 0);

	// Seeds with all five seed values
	__device__ __host__ inline void seed(
		unsigned int seedA, unsigned int seedB = 1, unsigned int seedC = 1, unsigned int seedD = 1, unsigned int seedE = 0);

	// Seeds with rand(:
	inline void seed();

	// Generates random unit
	__device__ __host__ inline unsigned int get();

	// Generates signed integer
	__device__ __host__ inline int getInt();

	// Unsigned range between minimum (inclusive) and maximum (exclusive) values
	__device__ __host__ inline unsigned int rangeUnsigned(unsigned int minimum, unsigned int maximum);

	// Signed range between minimum (inclusive) and maximum (exclusive) values
	__device__ __host__ inline unsigned int rangeSigned(int minimum, int maximum);

	// Random float between 0 (inclusive) and 1 (inclusive):
	__device__ __host__ inline float getFloat();

	// Random float between minimum (inclusive) and maximum (includive) values
	__device__ __host__ inline float range(float minimum, float maximum);

	// Returns random bool (chance is the chance of true)
	__device__ __host__ inline bool getBool(float chance);



private:
	static std::mutex lock;
	unsigned int a, b, c, d, e;
};



#include"DumbRand.impl.cuh"
