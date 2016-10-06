#pragma once

#include"Triangle.h"
#include"Stacktor.cuh"

struct BakedTriFace{
	Triangle vert, norm, tex;

	__device__ __host__ inline BakedTriFace();
	__device__ __host__ inline BakedTriFace(const Triangle &verts, const Triangle &norms, const Triangle &texs);
};

typedef Stacktor<BakedTriFace, 1> BakedTriMesh;





#include"BakedTriMesh.impl.h"

