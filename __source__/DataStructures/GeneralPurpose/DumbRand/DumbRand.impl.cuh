#include"DumbRand.cuh"


// Default constructor (does absolutely nothing)
__device__ __host__ inline DumbRand::DumbRand() {}

// Constructor that seeds all five seed values
__device__ __host__ inline DumbRand::DumbRand(
	UnsignedInt seedA, UnsignedInt seedB, UnsignedInt seedC, UnsignedInt seedD, UnsignedInt seedE) { 
	seed(seedA, seedB, seedC, seedD, seedE);
}

// Seeds with all five seed values
__device__ __host__ inline void DumbRand::seed(
	UnsignedInt seedA, UnsignedInt seedB, UnsignedInt seedC, UnsignedInt seedD, UnsignedInt seedE) {
	a = seedA; b = seedB; c = seedC; d = seedD, e = seedE;
}

// Seeds with rand()
inline void DumbRand::seed() {
	std::lock_guard<std::mutex> guard(lock);
	seed(rand(), rand(), rand(), rand(), rand());
}

// Generates random unit
__device__ __host__ inline DumbRand::UnsignedInt DumbRand::get() {
	// Basically... Copy-pasted from Wikipedia:
	register UnsignedInt s, t = d;
	t ^= t >> 2;
	t ^= t << 1;
	d = c; c = b; b = s = a;
	t ^= s;
	t ^= s << 4;
	a = t;
	return (t + (e += 362437));
}

// Generates signed integer
__device__ __host__ inline DumbRand::SignedInt DumbRand::getInt() {
	UnsignedInt value = get();
	return (*((SignedInt*)(&value)));
}

// Unsigned range between minimum (inclusive) and maximum (exclusive) values
__device__ __host__ inline DumbRand::UnsignedInt DumbRand::rangeUnsigned(UnsignedInt minimum, UnsignedInt maximum) {
	return ((get() % (maximum - minimum)) + minimum);
}

// Signed range between minimum (inclusive) and maximum (exclusive) values
__device__ __host__ inline DumbRand::UnsignedInt DumbRand::rangeSigned(SignedInt minimum, SignedInt maximum) {
	return ((getInt() % (maximum - minimum)) + minimum);
}

// Random float between 0 (inclusive) and 1 (inclusive):
__device__ __host__ inline float DumbRand::getFloat() {
	return (((float)get()) / ((float)UINT32_MAX));
}

// Random float between minimum (inclusive) and maximum (includive) values
__device__ __host__ inline float DumbRand::range(float minimum, float maximum) {
	return ((getFloat() * (maximum - minimum)) + minimum);
}
