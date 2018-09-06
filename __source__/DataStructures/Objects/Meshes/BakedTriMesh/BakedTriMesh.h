#pragma once

#include"../../../Primitives/Compound/Triangle/Triangle.h"
#include"../../Components/Transform/Transform.h"
#include"../../../GeneralPurpose/Stacktor/Stacktor.cuh"

struct BakedTriFace{
	Triangle vert, norm, tex;

	__device__ __host__ inline BakedTriFace();
	__device__ __host__ inline BakedTriFace(const Triangle &verts, const Triangle &norms, const Triangle &texs);
};

typedef Stacktor<BakedTriFace, 1> BakedTriMesh;


__device__ __host__ inline BakedTriFace operator>>(const BakedTriFace &face, const Transform &trans);
__device__ __host__ inline BakedTriFace& operator>>=(BakedTriFace &face, const Transform &trans);
__device__ __host__ inline BakedTriFace operator<<(const BakedTriFace &face, const Transform &trans);
__device__ __host__ inline BakedTriFace& operator<<=(BakedTriFace &face, const Transform &trans);



#include"BakedTriMesh.impl.h"

