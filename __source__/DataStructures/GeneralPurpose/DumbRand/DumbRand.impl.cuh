#include"DumbRand.cuh"


// Default constructor (does absolutely nothing)
__device__ __host__ inline DumbRand::DumbRand() {}

// Constructor that seeds all five seed values
__device__ __host__ inline DumbRand::DumbRand(
	unsigned int seedA, unsigned int seedB, unsigned int seedC, unsigned int seedD, unsigned int seedE) { 
	seed(seedA, seedB, seedC, seedD, seedE);
}

// Seeds with all five seed values
__device__ __host__ inline void DumbRand::seed(
	unsigned int seedA, unsigned int seedB, unsigned int seedC, unsigned int seedD, unsigned int seedE) {
	a = seedA; b = seedB; c = seedC; d = seedD, e = seedE;
}

// Seeds with rand()
inline void DumbRand::seed() {
	std::lock_guard<std::mutex> guard(lock);
	seed(rand(), rand(), rand(), rand(), rand());
}

// Generates random unit
__device__ __host__ inline unsigned int DumbRand::get() {
	// Basically... Copy-pasted from Wikipedia:
	register unsigned int s, t = d;
	t ^= t >> 2;
	t ^= t << 1;
	d = c; c = b; b = s = a;
	t ^= s;
	t ^= s << 4;
	a = t;
	return (t + (e += 362437));
}

// Generates signed integer
__device__ __host__ inline int DumbRand::getInt() {
	unsigned int value = get();
	return (*((int*)(&value)));
}

// Unsigned range between minimum (inclusive) and maximum (exclusive) values
__device__ __host__ inline unsigned int DumbRand::rangeUnsigned(unsigned int minimum, unsigned int maximum) {
	return ((get() % (maximum - minimum)) + minimum);
}

// Signed range between minimum (inclusive) and maximum (exclusive) values
__device__ __host__ inline unsigned int DumbRand::rangeSigned(int minimum, int maximum) {
	return ((((int)(get() >> 1)) % (maximum - minimum)) + minimum);
}

// Random float between 0 (inclusive) and 1 (inclusive):
__device__ __host__ inline float DumbRand::getFloat() {
	return (((float)get()) / ((float)UINT32_MAX));
}

// Random float between minimum (inclusive) and maximum (includive) values
__device__ __host__ inline float DumbRand::range(float minimum, float maximum) {
	return ((getFloat() * (maximum - minimum)) + minimum);
}

// Returns random bool (chance is the chance of true)
__device__ __host__ inline bool DumbRand::getBool(float chance) {
	return (getFloat() <= chance);
}

// Generates a point on a sphere of given radius:
__device__ __host__ inline void DumbRand::pointOnSphere(float &x, float &y, float &z, float radius) {
	float theta = (2.0f * 3.14159265359f * getFloat());
	float cosPhi = (1.0f - (2.0f * getFloat()));
	float sinPhi = sqrt(1.0f - (cosPhi * cosPhi));
	x = ((sinPhi * cos(theta)) * radius);
	y = ((sinPhi * sin(theta)) * radius);
	z = (cosPhi * radius);
}
