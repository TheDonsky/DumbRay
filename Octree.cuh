#pragma once
#include"PolyMesh.h"
#include"Material.cuh"
#include"AABB.h"
#include"MemManip.cuh"
#include"Shapes.cuh"


#define OCTREE_DEFAULT_SIZE Vector3(100000, 100000, 100000)
#define OCTREE_POLYCOUNT_TO_SPLIT_NODE 12
#define OCTREE_VOXEL_LOCAL_CAPACITY 12
#define OCTREE_MAX_DEPTH 16
//#define OCTREE_FILTER_NODES


template<typename ElemType>
class Octree;
template<typename ElemType>
class TypeTools<Octree<ElemType> >{
public:
	typedef Octree<ElemType> ElementType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(ElementType);
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
		TreeNode *children;

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
	__host__ inline Octree(AABB bounds = AABB(-OCTREE_DEFAULT_SIZE, OCTREE_DEFAULT_SIZE));
	__host__ inline void reinit(AABB bounds = AABB(-OCTREE_DEFAULT_SIZE, OCTREE_DEFAULT_SIZE));
	__host__ inline Octree(const Octree &octree);
	__host__ inline Octree& operator=(const Octree &octree);
	__host__ inline Octree(Octree &&octree);
	__host__ inline Octree& operator=(Octree &&octree);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/


	/** ========================================================== **/
	/*| push & build |*/
	__host__ inline void reset();
	__host__ inline void push(const Stacktor<ElemType> &objcets);
	__host__ inline void push(const ElemType &object);
	__host__ inline void build();
	__host__ inline void reduceNodes();

	
	/** ========================================================== **/
	/*| put |*/
	__host__ inline void put(const Stacktor<ElemType> &objcets);
	__host__ inline void put(const ElemType &elem);


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
	Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > nodeData;
	Stacktor<ElemType> data;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/

	/** ========================================================== **/
	__device__ __host__ inline void fixTreeNodePointers(const TreeNode *falseRoot);
	__device__ __host__ inline void fixNodeDataPointers(const ElemType *falseRoot);

	/** ========================================================== **/
	__device__ __host__ inline void pushData(const ElemType &object);
	__device__ __host__ inline void flushTree();

	/** ========================================================== **/
	__device__ __host__ __noinline__ void split(int index, int depth);
	__device__ __host__ inline bool splittingMakesSence(int index);
	__device__ __host__ inline void splitNode(int index, Vertex center);
	__device__ __host__ inline void reduceNode(int index);

	/** ========================================================== **/
	__device__ __host__ __noinline__ void put(const ElemType *elem, int nodeIndex, int depth);

	/** ========================================================== **/
	struct CastFrame{
		const TreeNode *node;
		char priorityChild;
		char curChild;
#ifdef OCTREE_FILTER_NODES
		char ignoreLow;
		char ignoreHigh;
#endif // OCTREE_FILTER_NODES
	};
	__device__ __host__ inline static void configureCastFrame(CastFrame &frame, const TreeNode *children, const Ray &r);
	__device__ __host__ inline bool castInLeaf(const Ray &r, RaycastHit &hit, int index, bool clipBackfaces)const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Friends: **/
	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(Octree);
};




#include"Octree.impl.cuh"

