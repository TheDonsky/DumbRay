#pragma once
#include "../Octree/Octree.cuh"
#include "../../../Components/Shaders/Material.cuh"
#include "../../../Components/Shaders/DefaultShader/DefaultShader.cuh"


template<typename HitType> struct Shaded;
template<typename HitType>
class TypeTools<Shaded<HitType> > {
public:
	typedef Shaded<HitType> ShadedType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(ShadedType);
};
template<typename HitType> struct ShadedOctree;
template<typename HitType>
class TypeTools<ShadedOctree<HitType> > {
public:
	typedef ShadedOctree<HitType> ShadedOctreeType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(ShadedOctreeType);
};


template<typename HitType>
struct Shaded {
	HitType object;
	Material<HitType> *material;

	__dumb__ Shaded();
	__dumb__ Shaded(const HitType &obj, Material<HitType> *mat);

	__dumb__ bool intersects(const Shaded &other)const;
	__dumb__ bool intersects(const AABB &other)const;
	__dumb__ bool cast(const Ray& r, bool clipBackface)const;
	__dumb__ bool castPreInversed(const Ray& inversedRay, bool clipBackface)const;
	__dumb__ bool cast(const Ray& ray, float &hitDistance, Vertex& hitPoint, bool clipBackface)const;
	template<typename BoundType>
	__dumb__ bool sharesPoint(const Shaded& b, const BoundType& commonPointBounds)const;
	template<typename Shape>
	__dumb__ Vertex intersectionCenter(const Shape &shape)const;
	template<typename Shape>
	__dumb__ AABB intersectionBounds(const Shape &shape)const;
	__dumb__ Vertex massCenter()const;
	__dumb__ AABB boundingBox()const;
	__dumb__ void dump()const;
};


template<typename HitType>
class ShadedOctree {
public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	// Creates a new Octree with given bounds
	__host__ inline ShadedOctree(AABB bounds = AABB(-OCTREE_DEFAULT_SIZE, OCTREE_DEFAULT_SIZE));
	// Recreates a new Octree with given bounds
	__host__ inline void reinit(AABB bounds = AABB(-OCTREE_DEFAULT_SIZE, OCTREE_DEFAULT_SIZE));
	// Copy-Constructor
	__host__ inline ShadedOctree(const ShadedOctree &octree);
	// Deep copy function
	__host__ inline ShadedOctree& operator=(const ShadedOctree &octree);
	// Steal constructor
	__host__ inline ShadedOctree(ShadedOctree &&octree);
	// Steal copy function
	__host__ inline ShadedOctree& operator=(ShadedOctree &&octree);
	// Swap function
	__host__ inline void swapWith(ShadedOctree &octree);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/


	/** ========================================================== **/
	/*| push & build |*/
	// Cleans Octree
	__host__ inline void reset();
	
	// Pushes list of objects (needs calling build as a final statement) (default material)
	__host__ inline void push(const Stacktor<HitType> &objects);
	// Pushes list of objects (needs calling build as a final statement) (returns true, if materialId is valid)
	__host__ inline bool push(const Stacktor<HitType> &objects, int materialId);
	// Pushes list of objects (needs calling build as a final statement) (returns material id)
	__host__ inline int push(const Stacktor<HitType> &objects, const Material<HitType> &material);
	
	// Pushes an object (needs calling build as a final statement) (default material)
	__host__ inline void push(const HitType &object);
	// Pushes an object (needs calling build as a final statement) (returns true, if materialId is valid)
	__host__ inline bool push(const HitType &object, int materialId);
	// Pushes an object (needs calling build as a final statement) (returns material id)
	__host__ inline int push(const HitType &object, const Material<HitType> &material);
	
	// Builds the Octree (needed if and only if it was filled with push() calls, not put()-s)
	__host__ inline void build();


	/** ========================================================== **/
	/*| put |*/
	// Adds the list of objects to the Octree (default material)
	__host__ inline void put(const Stacktor<HitType> &objects);
	// Adds the list of objects to the Octree (returns true, if materialId is valid)
	__host__ inline bool put(const Stacktor<HitType> &objects, int materialId);
	// Adds the list of objects to the Octree (returns material id)
	__host__ inline int put(const Stacktor<HitType> &objects, const Material<HitType> &material);

	// Adds the object to the Octree (default material)
	__host__ inline void put(const HitType &elem);
	// Adds the object to the Octree (returns true, if materialId is valid)
	__host__ inline bool put(const HitType &elem, int materialId);
	// Adds the object to the Octree (returns material id)
	__host__ inline int put(const HitType &elem, const Material<HitType> &material);
	
	// Optimizes tree nodes so that their sizes are no larger than they need to be for the current configuration
	// (Recommended after pushing/putting the entire scene in the Octree; after this function, addition will become unreliable)
	__host__ inline void reduceNodes();


	/** ========================================================== **/
	/*| cast |*/
	// Function, that lets the cast terminate
	typedef typename Octree<Shaded<HitType> >::CastBreaker CastBreaker;
	// Raycast output
	typedef typename Octree<Shaded<HitType> >::RaycastHit RaycastHit;
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
	__device__ __host__ inline Stacktor<Shaded<HitType> >& getData();
	// Returns data
	__device__ __host__ inline const Stacktor<Shaded<HitType> >& getData()const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	// Adds material (returns materialId)
	__host__ inline int addMaterial(const Material<HitType> &material);
	// Material count
	__device__ __host__ inline int materialCount()const;
	// Returns default material
	__device__ __host__ inline Material<HitType> &defaultMaterial();
	// Returns default material
	__device__ __host__ inline const Material<HitType> &defaultMaterial()const;
	// Returns material with given id (may crash, if id is not valid)
	__device__ __host__ inline Material<HitType> &material(int id);
	// Returns material with given id (may crash, if id is not valid)
	__device__ __host__ inline const Material<HitType> &material(int id)const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	DEFINE_CUDA_LOAD_INTERFACE_FOR(ShadedOctree);





private:
	Octree<Shaded<HitType> > octree;
	Stacktor<Material<HitType> > materials;

	__device__ __host__ inline void fixMaterialPointers(const Material<HitType> *falseRoot);

	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(ShadedOctree);
};





#include"ShadedOctree.impl.cuh"
