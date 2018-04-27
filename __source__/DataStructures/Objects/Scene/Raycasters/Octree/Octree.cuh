#pragma once
#include"../../../Meshes/PolyMesh/PolyMesh.h"
#include"../../../Components/Shaders/Material.cuh"
#include"../../../../Primitives/Compound/AABB/AABB.h"
#include"../../../../../Namespaces/MemManip/MemManip.cuh"
#include"../../../../../Namespaces/Shapes/Shapes.cuh"
#include"../Raycaster.cuh"


#define OCTREE_DEFAULT_SIZE Vector3(100000, 100000, 100000)
#define OCTREE_POLYCOUNT_TO_SPLIT_NODE 32
#define OCTREE_VOXEL_LOCAL_CAPACITY OCTREE_POLYCOUNT_TO_SPLIT_NODE
#define OCTREE_MAX_DEPTH 24


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
/*
	Octree our main data structure for accelerating raycasts.
*/
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

	typedef RaycastHit<ElemType> RaycastHit;




public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	// Creates a new Octree with given bounds
	__host__ inline Octree(AABB bounds = AABB(-OCTREE_DEFAULT_SIZE, OCTREE_DEFAULT_SIZE));
	// Recreates a new Octree with given bounds
	__host__ inline void reinit(AABB bounds = AABB(-OCTREE_DEFAULT_SIZE, OCTREE_DEFAULT_SIZE));
	// Copy-Constructor
	__host__ inline Octree(const Octree &octree);
	// Deep copy function
	__host__ inline Octree& operator=(const Octree &octree);
	// Steal constructor
	__host__ inline Octree(Octree &&octree);
	// Steal copy function
	__host__ inline Octree& operator=(Octree &&octree);
	// Swap function
	__host__ inline void swapWith(Octree &octree);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/


	/** ========================================================== **/
	/*| push & build |*/
	// Cleans Octree
	__host__ inline void reset();
	// Pushes list of objects (needs calling build as a final statement)
	__host__ inline void push(const Stacktor<ElemType> &objects);
	// Pushes an object (needs calling build as a final statement)
	__host__ inline void push(const ElemType &object);
	// Builds the Octree (needed if and only if it was filled with push() calls, not put()-s)
	__host__ inline void build();
	// Optimizes tree nodes so that their sizes are no larger than they need to be for the current configuration
	// (Recommended after pushing/putting the entire scene in the Octree; after this function, addition will become unreliable)
	__host__ inline void reduceNodes();

	
	/** ========================================================== **/
	/*| put |*/
	// Adds the list of objects to the Octree
	__host__ inline void put(const Stacktor<ElemType> &objects);
	// Adds the object to the Octree
	__host__ inline void put(const ElemType &elem);


	/** ========================================================== **/
	/*| cast |*/
	// Function, that lets the cast terminate
	typedef bool(*CastBreaker)(RaycastHit &hit, const Ray &ray, bool &rv);
	// Casts a ray and returns RaycastHit (if ray hits nothing, hitDistance will be set to FLT_MAX)
	__device__ __host__ inline RaycastHit cast(const Ray &r, bool clipBackfaces = true, CastBreaker castBreaker = NULL)const;
	// Casts a ray (returns true if the ray hits something; result is written in hit)
	__device__ __host__ inline bool cast(const Ray &r, RaycastHit &hit, bool clipBackfaces = true, CastBreaker castBreaker = NULL)const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	// Returns current node count
	__device__ __host__ inline int getNodeCount()const;
	// "Dumps the internals" in the console (or whatever's mapped on the standard output)
	__device__ __host__ inline void dump()const;
	// Returns data
	__device__ __host__ inline Stacktor<ElemType>& getData();
	// Returns data
	__device__ __host__ inline const Stacktor<ElemType>& getData()const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	DEFINE_CUDA_LOAD_INTERFACE_FOR(Octree);
	typedef const ElemType* ElemReference;





private:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	Stacktor<TreeNode> tree;
	Stacktor<Stacktor<ElemReference, OCTREE_VOXEL_LOCAL_CAPACITY> > nodeData;
	Stacktor<ElemType> data;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/

	/** ========================================================== **/
	__device__ __host__ inline void fixTreeNodePointers(const TreeNode *falseRoot);
	__device__ __host__ inline void fixNodeDataPointers(ElemReference falseRoot);

	/** ========================================================== **/
	__device__ __host__ inline void pushData(const ElemType &object);
	__device__ __host__ inline void flushTree();

	/** ========================================================== **/
	__device__ __host__ __noinline__ void split(int index, int depth);
	__device__ __host__ inline static void splitAABB(const AABB &aabb, const Vertex &center, AABB *result);
	__device__ __host__ inline bool splittingMakesSence(int index, const AABB *sub);
	__device__ __host__ inline void splitNode(int index, const AABB *sub);
	__device__ __host__ inline void reduceNode(int index);

	/** ========================================================== **/
	__device__ __host__ __noinline__ void put(ElemReference elem, int nodeIndex, int depth);

	/** ========================================================== **/
	struct CastFrame{
		const TreeNode *node;
		char priorityChild;
		char curChild;
	};
	__device__ __host__ inline static void configureCastFrame(CastFrame &frame, const TreeNode *children, const Ray &r);
	__device__ __host__ inline bool castInLeaf(const Ray &r, RaycastHit &hit, int index, bool clipBackfaces, CastBreaker castBreaker)const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Friends: **/
	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(Octree);
};




#include"Octree.impl.cuh"

