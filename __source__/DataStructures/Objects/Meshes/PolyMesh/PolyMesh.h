#pragma once

#include"../../../Primitives/Pure/Vector3/Vector3.h"
#include"../../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include"../BakedTriMesh/BakedTriMesh.h"


class PolyMesh;
SPECIALISE_TYPE_TOOLS_FOR(PolyMesh);





class PolyMesh{
public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	typedef Stacktor<Vector3, 1> VertexList;
	struct IndexNode{
		int vert, norm, tex;
		__device__ __host__ inline IndexNode();
		__device__ __host__ inline IndexNode(int v, int n, int t);
	};
	typedef Stacktor<IndexNode, 4> IndexFace;
	typedef Stacktor<IndexFace, 1> FaceList;
	struct BakedNode{
		Vector3 vert, norm, tex;
		__device__ __host__ inline BakedNode();
		__device__ __host__ inline BakedNode(Vertex v, Vector3 n, Vertex t);
	};
	typedef Stacktor<BakedNode, 4> Face;





public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	__device__ __host__ inline PolyMesh();
	__device__ __host__ inline PolyMesh(const VertexList &vertices, const VertexList &normals, const VertexList &textures, const FaceList &indexFaces);
	__device__ __host__ inline PolyMesh(const PolyMesh &m);
	__device__ __host__ inline PolyMesh& operator=(const PolyMesh &m);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	
	__device__ __host__ inline int vertextCount()const;
	__device__ __host__ inline Vertex& vertex(int index);
	__device__ __host__ inline const Vertex& vertex(int index)const;
	__device__ __host__ inline void addVertex(const Vertex &v);
	__device__ __host__ inline void removeVertex(int index);

	__device__ __host__ inline int normalCount()const;
	__device__ __host__ inline Vector3& normal(int index);
	__device__ __host__ inline const Vector3& normal(int index)const;
	__device__ __host__ inline void addNormal(const Vector3 &n);
	__device__ __host__ inline void removeNormal(int index);

	__device__ __host__ inline int textureCount()const;
	__device__ __host__ inline Vertex& texture(int index);
	__device__ __host__ inline const Vertex& texture(int index)const;
	__device__ __host__ inline void addTexture(const Vertex &t);
	__device__ __host__ inline void removeTexture(int index);

	__device__ __host__ inline int faceCount()const;
	__device__ __host__ inline IndexFace& indexFace(int index);
	__device__ __host__ inline const Face face(int index)const;
	__device__ __host__ inline const IndexFace& indexFace(int index)const;
	__device__ __host__ inline void addFace(const IndexFace &f);
	__device__ __host__ inline void addFace(const Face &f);
	__device__ __host__ inline void removeFace(int index);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	__device__ __host__ inline void bake(BakedTriMesh &mesh)const;
	__device__ __host__ inline BakedTriMesh bake()const;
	__device__ __host__ inline operator BakedTriMesh()const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	DEFINE_CUDA_LOAD_INTERFACE_FOR(PolyMesh);





private:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	struct Data{
		VertexList verts, norms, texs;
	}data;
	FaceList faces;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Friends: **/
	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(PolyMesh);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	__device__ __host__ inline PolyMesh& constructFrom(const VertexList &vertices, const VertexList &normals, const VertexList &textures, const FaceList &indexFaces);
	
	__device__ __host__ inline static int getVert(IndexNode &node);
	__device__ __host__ inline static void setVert(IndexNode &node, int value);
	__device__ __host__ inline static int getNorm(IndexNode &node);
	__device__ __host__ inline static void setNorm(IndexNode &node, int value);
	__device__ __host__ inline static int getTex(IndexNode &node);
	__device__ __host__ inline static void setTex(IndexNode &node, int value);
	__device__ __host__ inline static void removeVert(VertexList &vertData, int index, FaceList &faceBuffer, int(*indexGet)(IndexNode&), void(*indexSet)(IndexNode&, int));
};





#include"PolyMesh.impl.h"
