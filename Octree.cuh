#pragma once
#include"PolyMesh.h"
#include"Material.cuh"
#include"AABB.h"
#include"MemManip.cuh"
#include"Shapes.cuh"


#define OCTREE_DEFAULT_SIZE Vector3(100000, 100000, 100000)
#define OCTREE_POLYCOUNT_TO_SPLIT_NODE 16
#define OCTREE_VOXEL_LOCAL_CAPACITY 16
#define OCTREE_MAX_DEPTH 16


template<typename ElemType>
class Octree;
template<typename ElemType>
class StacktorTypeTools<Octree<ElemType> >{
public:
	typedef Octree<ElemType> ElementType;
	DEFINE_STACKTOR_TYPE_TOOLS_CONTENT_FOR(ElementType);
};





template<typename ElemType  = BakedTriFace>
/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** ########################################################################## **/
/** Octree:                                                                    **/
/** ########################################################################## **/
class Octree{
public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/


	/** ========================================================== **/
	struct TreeNode{
		AABB bounds;
		bool hasChildren;
		TreeNode *children[8];

		__device__ __host__ inline TreeNode(AABB boundingBox = AABB());
	};


	/** ========================================================== **/
	struct RaycastHit{
		ElemType object;
		float hitDistance;
		Vector3 hitPoint;

		__device__ __host__ inline RaycastHit();
		__device__ __host__ inline RaycastHit(const ElemType &elem, const float d, const Vector3 &p);
		__device__ __host__ inline RaycastHit& operator()(const ElemType &elem, const float d, const Vector3 &p);
		__device__ __host__ inline void set(const ElemType &elem, const float d, const Vector3 &p);
	};





public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	__device__ __host__ inline Octree(AABB bounds = AABB(-OCTREE_DEFAULT_SIZE, OCTREE_DEFAULT_SIZE));
	__device__ __host__ inline void reinit(AABB bounds = AABB(-OCTREE_DEFAULT_SIZE, OCTREE_DEFAULT_SIZE));





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/


	/** ========================================================== **/
	/*| push & build |*/
	__device__ __host__ inline void reset();
	__device__ __host__ inline void push(const Stacktor<ElemType> &objcets);
	__device__ __host__ inline void push(const ElemType &object);
	__device__ __host__ inline void build();

	
	/** ========================================================== **/
	/*| put |*/
	__device__ __host__ inline void put(const Stacktor<ElemType> &objcets);
	__device__ __host__ inline void put(const ElemType &elem);


	/** ========================================================== **/
	/*| cast |*/
	__device__ __host__ inline RaycastHit cast(const Ray &r, bool clipBackfaces = true)const;
	__device__ __host__ inline bool cast(const Ray &r, RaycastHit &hit, bool clipBackfaces = true)const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	__device__ __host__ inline int getNodeCount()const;
	__device__ __host__ inline void dump()const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	DEFINE_CUDA_LOAD_INTERFACE_FOR(Octree);





private:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	Stacktor<TreeNode> tree;
	Stacktor<Stacktor<int, OCTREE_VOXEL_LOCAL_CAPACITY> > nodeData;
	Stacktor<ElemType> data;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/

	/** ========================================================== **/
	__device__ __host__ inline void flushTree();
	__device__ __host__ __noinline__ void split(int index, int depth);
	__device__ __host__ inline bool splittingMakesSence(int index);
	__device__ __host__ inline void splitNode(int index, Vertex center);
	__device__ __host__ inline void reduceNodes();
	__device__ __host__ inline void reduceNode(int index);

	/** ========================================================== **/
	__device__ __host__ __noinline__ void put(const ElemType &elem, int nodeIndex, int dataIndex, int depth);

	/** ========================================================== **/
	struct CastFrame{
		const TreeNode *node;
		char priorityChild;
		char curChild;
	};
	__device__ __host__ inline bool castInLeaf(const Ray &r, RaycastHit &hit, int index, bool clipBackfaces)const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Friends: **/
	DEFINE_STACKTOR_TYPE_TOOLS_FRIENDSHIP_FOR(Octree);
};




#include"Octree.impl.cuh"

