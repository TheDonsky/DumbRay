#pragma once
#include"PolyMesh.h"



namespace Primitives {
	__device__ __host__ inline PolyMesh sphere(int edges = 32, float radius = 1);
}





#include"primitives.impl.cuh"
