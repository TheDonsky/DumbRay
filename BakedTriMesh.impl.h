#include"BakedTriMesh.h"



__device__ __host__ inline BakedTriFace::BakedTriFace(){ }
__device__ __host__ inline BakedTriFace::BakedTriFace(const Triangle &verts, const Triangle &norms, const Triangle &texs){
	vert = verts;
	norm = norms;
	tex = texs;
}
