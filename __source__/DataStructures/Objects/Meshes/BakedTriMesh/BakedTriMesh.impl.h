#include"BakedTriMesh.h"



__device__ __host__ inline BakedTriFace::BakedTriFace(){ }
__device__ __host__ inline BakedTriFace::BakedTriFace(const Triangle &verts, const Triangle &norms, const Triangle &texs){
	vert = verts;
	norm = norms;
	tex = texs;
}





__device__ __host__ inline BakedTriFace operator>>(BakedTriFace &face, const Transform &trans) {
	return BakedTriFace(face.vert >> trans, (face.norm >> trans) - trans.getPosition(), face.tex);
}
__device__ __host__ inline BakedTriFace& operator>>=(BakedTriFace &face, const Transform &trans) {
	return (face = (face >> trans));
}
__device__ __host__ inline BakedTriFace operator<<(BakedTriFace &face, const Transform &trans) {
	return BakedTriFace(face.vert << trans, (face.norm + trans.getPosition()) << trans, face.tex);
}
__device__ __host__ inline BakedTriFace& operator<<=(BakedTriFace &face, const Transform &trans) {
	return (face = (face << trans));
}
