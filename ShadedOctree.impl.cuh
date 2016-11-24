#include "ShadedOctree.cuh"
#include "Shapes.cuh"

/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/

/** ========================================================== **/
template<typename HitType>
__device__ __host__ inline void TypeTools<Shaded<HitType> >::init(Shaded<HitType> &m) {
	TypeTools<HitType>::init(m.object);
	m.material = NULL;
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Shaded<HitType> >::dispose(Shaded<HitType> &m) {
	TypeTools<HitType>::dispose(m.object);
	m.material = NULL;
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Shaded<HitType> >::swap(Shaded<HitType> &a, Shaded<HitType> &b) {
	TypeTools<HitType>::swap(a.object, b.object);
	TypeTools<Material<HitType>*>::swap(a.material, b.material);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Shaded<HitType> >::transfer(Shaded<HitType> &src, Shaded<HitType> &dst) { 
	TypeTools<HitType>::transfer(src.object, dst.object);
	TypeTools<Material<HitType>*>::transfer(src.material, dst.material);
}

template<typename HitType>
inline bool TypeTools<Shaded<HitType> >::prepareForCpyLoad(const Shaded<HitType> *source, Shaded<HitType> *hosClone, Shaded<HitType> *devTarget, int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if (!TypeTools<HitType>::prepareForCpyLoad(&source[i].object, &hosClone[i].object, &((devTarget + i)->object), 1)) break;
		if (!TypeTools<Material<HitType>*>::prepareForCpyLoad(&source[i].material, &hosClone[i].material, &((devTarget + i)->material), 1)) {
			TypeTools<HitType>::undoCpyLoadPreparations(&source[i].object, &hosClone[i].object, &((devTarget + i)->object), 1);
			break;
		}
	}
	if (i < count) {
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return(false);
	}
	return(true);
}

template<typename HitType>
inline void TypeTools<Shaded<HitType> >::undoCpyLoadPreparations(const Shaded<HitType> *source, Shaded<HitType> *hosClone, Shaded<HitType> *devTarget, int count) {
	for (int i = 0; i < count; i++) {
		TypeTools<HitType>::undoCpyLoadPreparations(&source[i].object, &hosClone[i].object, &((devTarget + i)->object), 1);
		TypeTools<Material<HitType>*>::undoCpyLoadPreparations(&source[i].material, &hosClone[i].material, &((devTarget + i)->material), 1);
	}
}
template<typename HitType>
inline bool TypeTools<Shaded<HitType> >::devArrayNeedsToBeDisoposed() { 
	return (TypeTools<HitType>::devArrayNeedsToBeDisoposed() || TypeTools<Material<HitType>*>::devArrayNeedsToBeDisoposed());
}
template<typename HitType>
inline bool TypeTools<Shaded<HitType> >::disposeDevArray(Shaded<HitType> *arr, int count) {
	for (int i = 0; i < count; i++) {
		if (!TypeTools<HitType>::disposeDevArray(&((arr + i)->object), 1)) return false;
		if (!TypeTools<Material<HitType>*>::disposeDevArray(&((arr + i)->material), 1)) return false;
	}
	return true;
}


/** ========================================================== **/
template<typename HitType>
__device__ __host__ inline void TypeTools<ShadedOctree<HitType> >::init(ShadedOctree<HitType> &m) {
	TypeTools<Octree<Shaded<HitType> > >::init(m.octree);
	TypeTools<Stacktor<Material<HitType> > >::init(m.materials);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<ShadedOctree<HitType> >::dispose(ShadedOctree<HitType> &m) {
	TypeTools<Octree<Shaded<HitType> > >::dispose(m.octree);
	TypeTools<Stacktor<Material<HitType> > >::dispose(m.materials);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<ShadedOctree<HitType> >::swap(ShadedOctree<HitType> &a, ShadedOctree<HitType> &b) {
	a.swapWith(b);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<ShadedOctree<HitType> >::transfer(ShadedOctree<HitType> &src, ShadedOctree<HitType> &dst) {
	
	TypeTools<Octree<Shaded<HitType> > >::transfer(src.octree, dst.octree);
	TypeTools<Stacktor<Material<HitType> > >::transfer(src.materials, dst.materials);
}

namespace ShadedOctreePrivateKernels {
#define SHADED_OCTREE_PRIVATE_KERNELS_THREADS_PER_BLOCK 256
#define SHADED_OCTREE_PRIVATE_KERNELS_UNITS_PER_THREAD 16

	__device__ __host__ inline static int numThreads() {
		return SHADED_OCTREE_PRIVATE_KERNELS_THREADS_PER_BLOCK;
	}
	__device__ __host__ inline static int unitsPerThread() {
		return SHADED_OCTREE_PRIVATE_KERNELS_UNITS_PER_THREAD;
	}
	__device__ __host__ inline static int unitsPerBlock() {
		return (SHADED_OCTREE_PRIVATE_KERNELS_THREADS_PER_BLOCK * SHADED_OCTREE_PRIVATE_KERNELS_UNITS_PER_THREAD);
	}
	__device__ __host__ inline static int numBlocks(int totalNumber) {
		int perBlock = unitsPerBlock();
		return ((totalNumber + perBlock - 1) / perBlock);
	}

#undef SHADED_OCTREE_PRIVATE_KERNELS_THREADS_PER_BLOCK
#undef SHADED_OCTREE_PRIVATE_KERNELS_UNITS_PER_THREAD

	template<typename HitType>
	__global__ static void fixMaterialPointers(Material<HitType> *falseRoot, Material<HitType> *trueRoot, Shaded<HitType> *data, int count) {
		int start = (blockIdx.x * unitsPerBlock() + threadIdx.x * unitsPerThread());
		int end = start + unitsPerThread();
		if (end > count) end = count;
		Shaded<HitType> *endNode = (data + end);
		for (Shaded<HitType> *ptr = (data + start); ptr < endNode; ptr++)
			if (ptr->material != NULL) ptr->material = trueRoot + (ptr->material - falseRoot);
	}
}

template<typename HitType>
inline bool TypeTools<ShadedOctree<HitType> >::prepareForCpyLoad(const ShadedOctree<HitType> *source, ShadedOctree<HitType> *hosClone, ShadedOctree<HitType> *devTarget, int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if (!TypeTools<Octree<Shaded<HitType> > >::prepareForCpyLoad(&(source[i].octree), &(hosClone[i].octree), &((devTarget + i)->octree), 1)) break;
		if (!TypeTools<Stacktor<Material<HitType> > >::prepareForCpyLoad(&(source[i].materials), &(hosClone[i].materials), &((devTarget + i)->materials), 1)) {
			TypeTools<Octree<Shaded<HitType> > >::undoCpyLoadPreparations(&(source[i].octree), &(hosClone[i].octree), &((devTarget + i)->octree), 1);
			break;
		}
		bool streamError = false;
		cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) streamError = true;
		else {
			Material<HitType> *falseRoot = (source[i].materials + 0);
			Material<HitType> *trueRoot = (hosClone[i].materials + 0);
			Shaded<HitType> *data = (hosClone[i].octree.getData() + 0);
			int dataSize = hosClone[i].octree.getData().size();
			int kernelWidth = ShadedOctreePrivateKernels::numBlocks(dataSize);
			int kernelHeight = ShadedOctreePrivateKernels::numThreads();
			ShadedOctreePrivateKernels::fixMaterialPointers<HitType><<<kernelWidth, kernelHeight, 0, stream>>>(falseRoot, trueRoot, data, dataSize);
			streamError = (cudaStreamSynchronize(stream) != cudaSuccess);
			if (cudaStreamDestroy(stream) != cudaSuccess) streamError = true;
		}
		if (streamError) {
			TypeTools<Octree<Shaded<HitType> > >::undoCpyLoadPreparations(&(source[i].octree), &(hosClone[i].octree), &((devTarget + i)->octree), 1);
			TypeTools<Stacktor<Material<HitType> > >::undoCpyLoadPreparations(&(source[i].materials), &(hosClone[i].materials), &((devTarget + i)->materials), 1);
			break;
		}
	}
	if (i < count) {
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return(false);
	}
	return(true);
}

template<typename HitType>
inline void TypeTools<ShadedOctree<HitType> >::undoCpyLoadPreparations(const ShadedOctree<HitType> *source, ShadedOctree<HitType> *hosClone, ShadedOctree<HitType> *devTarget, int count) {
	for (int i = 0; i < count; i++) {
		TypeTools<Octree<Shaded<HitType> > >::undoCpyLoadPreparations(&(source[i].octree), &(hosClone[i].octree), &((devTarget + i)->octree), 1);
		TypeTools<Stacktor<Material<HitType> > >::undoCpyLoadPreparations(&(source[i].materials), &(hosClone[i].materials), &((devTarget + i)->materials), 1);
	}
}
template<typename HitType>
inline bool TypeTools<ShadedOctree<HitType> >::devArrayNeedsToBeDisoposed() {
	return (TypeTools<Octree<Shaded<HitType> > >::devArrayNeedsToBeDisoposed() || TypeTools<Stacktor<Material<HitType> > >::devArrayNeedsToBeDisoposed());
}
template<typename HitType>
inline bool TypeTools<ShadedOctree<HitType> >::disposeDevArray(ShadedOctree<HitType> *arr, int count) {
	for (int i = 0; i < count; i++) {
		if (!TypeTools<Octree<Shaded<HitType> > >::disposeDevArray(&((arr + i)->octree), 1)) return false;
		if (!TypeTools<Stacktor<Material<HitType> > >::disposeDevArray(&((arr + i)->materials), 1)) return false;
	}
	return true;
}



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
__dumb__ Shaded<HitType>::Shaded() { }
template<typename HitType>
__dumb__ Shaded<HitType>::Shaded(const HitType &obj, Material<HitType> *mat) {
	object = obj;
	material = mat;
}

template<typename HitType>
__dumb__ bool Shaded<HitType>::intersects(const Shaded &other)const {
	return Shapes::intersect<HitType>(object, other.object);
}
template<typename HitType>
__dumb__ bool Shaded<HitType>::cast(const Ray& r, bool clipBackface)const {
	return Shapes::cast<HitType>(r, object, clipBackface);
}
template<typename HitType>
__dumb__ bool Shaded<HitType>::castPreInversed(const Ray& inversedRay, bool clipBackface)const {
	return Shapes::castPreInversed<HitType>(inversedRay, object, clipBackface);
}
template<typename HitType>
__dumb__ bool Shaded<HitType>::cast(const Ray& ray, float &hitDistance, Vertex& hitPoint, bool clipBackface)const {
	float dist;
	Vertex hit;
	if (Shapes::cast<HitType>(ray, object, dist, hitPoint, clipBackface)) {
		if (material != NULL) {
			// Add transparency checking here...
		}
		hitDistance = dist;
		hitPoint = hit;
		return true;
	}
	else return false;
}
template<typename HitType>
template<typename BoundType>
__dumb__ bool Shaded<HitType>::sharesPoint(const Shaded& b, const BoundType& commonPointBounds)const {
	return Shapes::sharePoint<HitType, BoundType>(object, b.object, commonPointBounds);
}
template<typename HitType>
__dumb__ Vertex Shaded<HitType>::massCenter()const {
	return Shapes::massCenter<HitType>(object);
}
template<typename HitType>
__dumb__ AABB Shaded<HitType>::boundingBox()const {
	return Shapes::boundingBox<HitType>(object);
}
template<typename HitType>
__dumb__ void Shaded<HitType>::dump()const {
	printf("-----------------------------------\n");
	printf("Shaded Object: {\n");
	printf("Object: \n");
	Shapes::dump<HitType>(object);
	printf("\nMaterial: \n");
	if (material == NULL) printf("\tNULL");
	else Shapes::dump<Material<HitType> >(*material);
	printf("\n}\n");
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
/* Uploads unit to CUDA device and returns the clone address */
inline ShadedOctree<HitType>* ShadedOctree<HitType>::upload()const {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_BODY(ShadedOctree);
}
template<typename HitType>
/* Uploads unit to the given location on the CUDA device (returns true, if successful; needs RAW data address) */
inline bool ShadedOctree<HitType>::uploadAt(ShadedOctree *address)const {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_AT_BODY(ShadedOctree);
}
template<typename HitType>
/* Uploads given source array/unit to the given target location on CUDA device (returns true, if successful; needs RAW data address) */
inline bool ShadedOctree<HitType>::upload(const ShadedOctree *source, ShadedOctree *target, int count) {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_AT_BODY(ShadedOctree);
}
template<typename HitType>
/* Uploads given source array/unit to CUDA device and returns the clone address */
inline ShadedOctree<HitType>* ShadedOctree<HitType>::upload(const ShadedOctree<HitType> *source, int count) {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_BODY(ShadedOctree);
}
template<typename HitType>
/* Disposed given array/unit on CUDA device, making it ready to be free-ed (returns true, if successful) */
inline bool ShadedOctree<HitType>::dispose(ShadedOctree *arr, int count) {
	IMPLEMENT_CUDA_LOAD_INTERFACE_DISPOSE_BODY(ShadedOctree);
}







/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
// Creates a new Octree with given bounds
__host__ inline ShadedOctree<HitType>::ShadedOctree(AABB bounds) {
	reinit(bounds);
}
template<typename HitType>
// Recreates a new Octree with given bounds
__host__ inline void ShadedOctree<HitType>::reinit(AABB bounds) {
	octree.reinit(bounds);
	materials.clear();
	materials.flush(2);
	materials[0].use<DefaultShader>();
}
template<typename HitType>
// Copy-Constructor
__host__ inline ShadedOctree<HitType>::ShadedOctree(const ShadedOctree &octree) {
	(*this) = octree;
}
template<typename HitType>
// Deep copy function
__host__ inline ShadedOctree<HitType>& ShadedOctree<HitType>::operator=(const ShadedOctree &octree) {
	if (this == (&octree)) return (*this);
	this->octree = octree.octree;
	materials = octree.materials;
	fixMaterialPointers(octree.materials + 0);
	return (*this);
}
template<typename HitType>
// Steal constructor
__host__ inline ShadedOctree<HitType>::ShadedOctree(ShadedOctree &&octree) : ShadedOctree() {
	swapWith(octree);
}
template<typename HitType>
// Steal copy function
__host__ inline ShadedOctree<HitType>& ShadedOctree<HitType>::operator=(ShadedOctree &&octree) {
	swapWith(octree);
	return (*this);
}
template<typename HitType>
// Swap function
__host__ inline void ShadedOctree<HitType>::swapWith(ShadedOctree &octree) {
	if (this == (&octree)) return;
	Material<HitType> *thisOldRoot = (materials + 0);
	Material<HitType> *otherOldRoot = (octree.materials + 0);
	this->octree.swapWith(octree.octree);
	materials.swapWith(octree.materials);
	fixMaterialPointers(otherOldRoot);
	octree.fixMaterialPointers(thisOldRoot);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/


/** ========================================================== **/
/*| push & build |*/
template<typename HitType>
// Cleans Octree
__host__ inline void ShadedOctree<HitType>::reset() {
	octree.reset();
	materials.clear();
	materials.flush(2);
	materials[0].use<DefaultShader>();
}
template<typename HitType>
// Adds material (returns materialId)
__host__ inline int ShadedOctree<HitType>::addMaterial(const Material &material) {
	int materialId = materials.size();
	Material<HitType> *materialRoot = (materials + 1);
	materials.push(material);
	fixMaterialPointers(materialRoot);
	return materialId;
}


template<typename HitType>
// Pushes list of objects (needs calling build as a final statement) (default material)
__host__ inline void ShadedOctree<HitType>::push(const Stacktor<HitType> &objects) {
	push(objects, 0);
}
template<typename HitType>
// Pushes list of objects (needs calling build as a final statement) (returns true, if materialId is valid)
__host__ inline bool ShadedOctree<HitType>::push(const Stacktor<HitType> &objects, int materialId) {
	if (materialId < 0 || materialId >= materials.size()) return false;
	for (int i = 0; i < objects.size(); i++)
		push(objects[i], materialId);
	return true;
}
template<typename HitType>
// Pushes list of objects (needs calling build as a final statement) (returns material id)
__host__ inline int ShadedOctree<HitType>::push(const Stacktor<HitType> &objects, const Material<HitType> &material) {
	int materialId = addMaterial(material);
	push(objects, materialId);
	return materialId;
}

template<typename HitType>
// Pushes an object (needs calling build as a final statement) (default material)
__host__ inline void ShadedOctree<HitType>::push(const HitType &object) {
	push(object, 0);
}
template<typename HitType>
// Pushes an object (needs calling build as a final statement) (returns true, if materialId is valid)
__host__ inline bool ShadedOctree<HitType>::push(const HitType &object, int materialId) {
	if (materialId < 0 || materialId >= materials.size()) return false;
	octree.push(Shaded<HitType>(object, materials + materialId));
	return true;
}
template<typename HitType>
// Pushes an object (needs calling build as a final statement) (returns material id)
__host__ inline int ShadedOctree<HitType>::push(const HitType &object, const Material<HitType> &material) {
	int materialId = addMaterial(material);
	push(object, materialId);
	return materialId;
}

template<typename HitType>
// Builds the Octree (needed if and only if it was filled with push() calls, not put()-s)
__host__ inline void ShadedOctree<HitType>::build();


/** ========================================================== **/
/*| put |*/
template<typename HitType>
// Adds the list of objects to the Octree (default material)
__host__ inline void ShadedOctree<HitType>::put(const Stacktor<HitType> &objects) {
	put(objects, 0);
}
template<typename HitType>
// Adds the list of objects to the Octree (returns true, if materialId is valid)
__host__ inline bool ShadedOctree<HitType>::put(const Stacktor<HitType> &objects, int materialId) {
	if (materialId < 0 || materialId >= materials.size()) return false;
	for (int i = 0; i < objects.size(); i++)
		put(objects[i], materialId);
	return true;
}
template<typename HitType>
// Adds the list of objects to the Octree (returns material id)
__host__ inline int ShadedOctree<HitType>::put(const Stacktor<HitType> &objects, const Material<HitType> &material) {
	int materialId = addMaterial(material);
	put(objects, materialId);
	return materialId;
}

template<typename HitType>
// Adds the object to the Octree (default material)
__host__ inline void ShadedOctree<HitType>::put(const HitType &elem) {
	put(elem, 0);
}
template<typename HitType>
// Adds the object to the Octree (returns true, if materialId is valid)
__host__ inline bool ShadedOctree<HitType>::put(const HitType &elem, int materialId) {
	if (materialId < 0 || materialId >= materials.size()) return false;
	octree.put(Shaded<HitType>(object, materials + materialId));
	return true;
}
template<typename HitType>
// Adds the object to the Octree (returns material id)
__host__ inline int ShadedOctree<HitType>::put(const HitType &elem, const Material<HitType> &material) {
	int materialId = addMaterial(material);
	put(object, materialId);
	return materialId;
}

template<typename HitType>
// Optimizes tree nodes so that their sizes are no larger than they need to be for the current configuration
// (Recommended after pushing/putting the entire scene in the Octree; after this function, addition will become unreliable)
__host__ inline void ShadedOctree<HitType>::reduceNodes() {
	octree.reduceNodes();
}


/** ========================================================== **/
/*| cast |*/
template<typename HitType>
// Casts a ray and returns RaycastHit (if ray hits nothing, hitDistance will be set to FLT_MAX)
__device__ __host__ inline ShadedOctree<HitType>::RaycastHit ShadedOctree<HitType>::cast(const Ray &r, bool clipBackfaces, CastBreaker castBreaker)const {
	return octree.cast(r, clipBackfaces, castBreaker);
}
template<typename HitType>
// Casts a ray (returns true if the ray hits something; result is written in hit)
__device__ __host__ inline bool ShadedOctree<HitType>::cast(const Ray &r, RaycastHit &hit, bool clipBackfaces, CastBreaker castBreaker)const {
	return octree.cast(r, hit, clipBackfaces, castBreaker);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
// Returns current node count
__device__ __host__ inline int ShadedOctree<HitType>::getNodeCount()const {
	return octree.getNodeCount();
}
template<typename HitType>
// "Dumps the internals" in the console (or whatever's mapped on the standard output)
__device__ __host__ inline void ShadedOctree<HitType>::dump()const {
	octree.dump();
}
template<typename HitType>
// Returns data
__device__ __host__ inline Stacktor<Shaded<HitType> >& ShadedOctree<HitType>::getData() {
	return octree.getData();
}
template<typename HitType>
// Returns data
__device__ __host__ inline const Stacktor<Shaded<HitType> >& ShadedOctree<HitType>::getData()const {
	return octree.getData();
}





template<typename HitType>
__device__ __host__ inline void ShadedOctree<HitType>::fixMaterialPointers(Material<HitType> *falseRoot){
	Material<HitType> *realRoot = (materials + 0);
	if (realRoot != falseRoot) {
		Stacktor<Shaded<HitType> > &data = getData();
		for (int i = 0; i < data.size(); i++)
			if (data[i].material != NULL)
				data[i].material = (realRoot + (data[i].material - falseRoot));
	}
}
