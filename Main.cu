#include "Vector3.h"
#include "Cutex.test.cuh"


__global__ static void doVectorShit() {
	Vector3 a;
	Vector3 b;
	Vector3 c = a + b;
}


static void doShit() {
	Vector3 a;
	Vector3 b;
	Vector3 c = a + b;
	doVectorShit<<<1, 1>>>();
	CutexTest::test();
}
