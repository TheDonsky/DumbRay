#include"Octree.cuh"

#define OCTREE_AABB_EPSILON_MULTIPLIER 64
#define OCTREE_AABB_EPSILON (OCTREE_AABB_EPSILON_MULTIPLIER * VECTOR_EPSILON)
#define OCTREE_AABB_EPSILON_VECTOR (EPSILON_VECTOR * OCTREE_AABB_EPSILON_MULTIPLIER)


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename ElemType>
// Creates a new Octree with given bounds
__host__ inline Octree<ElemType>::Octree(AABB bounds){
	reinit(bounds);
}
template<typename ElemType>
// Recreates a new Octree with given bounds
__host__ inline void Octree<ElemType>::reinit(AABB bounds){
	tree.clear();
	nodeData.clear();
	data.clear();
	tree.push(TreeNode(AABB(bounds.getMin() - OCTREE_AABB_EPSILON_VECTOR, bounds.getMax() + OCTREE_AABB_EPSILON_VECTOR)));
	nodeData.flush(1);
}
template<typename ElemType>
// Copy-Constructor
__host__ inline Octree<ElemType>::Octree(const Octree &octree) {
	(*this) = octree;
}
template<typename ElemType>
// Deep copy function
__host__ inline Octree<ElemType>& Octree<ElemType>::operator=(const Octree &octree) {
	tree = octree.tree;
	fixTreeNodePointers(octree.tree + 0);
	nodeData = octree.nodeData;
	data = octree.data;
	fixNodeDataPointers(octree.data + 0);
	return (*this);
}
template<typename ElemType>
// Steal constructor
__host__ inline Octree<ElemType>::Octree(Octree &&octree) {
	swapWith(octree);
}
template<typename ElemType>
// Steal copy function
__host__ inline Octree<ElemType>& Octree<ElemType>::operator=(Octree &&octree) {
	swapWith(octree);
	return (*this);
}
template<typename ElemType>
// Swap function
__host__ inline void  Octree<ElemType>::swapWith(Octree &octree) {
	const TreeNode *oldTreeRoot = (octree.tree + 0);
	const TreeNode *otherTreeRoot = (tree + 0);
	tree.swapWith(octree.tree);
	fixTreeNodePointers(oldTreeRoot);
	octree.fixTreeNodePointers(otherTreeRoot);
	const ElemType *oldDataRoot = (octree.data + 0);
	const ElemType *otherDataRoot = (data + 0);
	nodeData.swapWith(octree.nodeData);
	data.swapWith(octree.data);
	fixNodeDataPointers(oldDataRoot);
	octree.fixNodeDataPointers(otherDataRoot);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/


/** ========================================================== **/
/*| push & build |*/
template<typename ElemType>
// Cleans Octree
__host__ inline void Octree<ElemType>::reset(){
	tree.clear();
	nodeData.clear();
	data.clear();
	tree.flush(1);
	nodeData.flush(1);
}
template<typename ElemType>
// Pushes list of objects (needs calling build as a final statement)
__host__ inline void Octree<ElemType>::push(const Stacktor<ElemType> &objcets){
	for (int i = 0; i < objcets.size(); i++)
		push(objcets[i]);
}
template<typename ElemType>
// Pushes an object (needs calling build as a final statement)
__host__ inline void Octree<ElemType>::push(const ElemType &object){
	AABB box = Shapes::boundingBox<ElemType>(object);
	if (data.empty()) tree[0].bounds = box;
	else{
		const Vertex oldStart = tree[0].bounds.getMin();
		const Vertex oldEnd = tree[0].bounds.getMax();
		const Vertex boxStart = box.getMin();
		const Vertex boxEnd = box.getMax();
		Vertex newStart(min(oldStart.x, boxStart.x), min(oldStart.y, boxStart.y), min(oldStart.z, boxStart.z));
		Vertex newEnd(max(oldEnd.x, boxEnd.x), max(oldEnd.y, boxEnd.y), max(oldEnd.z, boxEnd.z));
		tree[0].bounds = AABB(newStart, newEnd);
	}
	pushData(object);
	nodeData[0].push(data + data.size() - 1);
}
template<typename ElemType>
// Builds the Octree (needed if and only if it was filled with push() calls, not put()-s)
__host__ inline void Octree<ElemType>::build(){
	split(0, 0);
	reduceNodes();
}
template<typename ElemType>
// Optimizes tree nodes so that their sizes are no larger than they need to be for the current configuration
// (Recommended after pushing/putting the entire scene in the Octree; after this function, addition will become unreliable)
__host__ inline void Octree<ElemType>::reduceNodes() {
	reduceNode(0);
}


/** ========================================================== **/
/*| put |*/
template<typename ElemType>
// Adds the list of objects to the Octree
__host__ inline void Octree<ElemType>::put(const Stacktor<ElemType> &objcets){
	for (int i = 0; i < objcets.size(); i++)
		put(objcets[i]);
}
template<typename ElemType>
// Adds the object to the Octree
__host__ inline void Octree<ElemType>::put(const ElemType &object){
	int dataIndex = data.size();
	pushData(object);
	put(data + dataIndex, 0, 0);
}


/** ========================================================== **/
/*| cast |*/
template<typename ElemType>
// Casts a ray and returns RaycastHit (if ray hits nothing, hitDistance will be set to FLT_MAX)
__device__ __host__ inline Octree<ElemType>::RaycastHit Octree<ElemType>::cast(const Ray &r, bool clipBackfaces)const{
	RaycastHit hit;
	if (!cast(r, hit, clipBackfaces))
		hit.hitDistance = FLT_MAX;
	return hit;
}
template<typename ElemType>
// Casts a ray (returns true if the ray hits something; result is written in hit)
__device__ __host__ inline bool Octree<ElemType>::cast(const Ray &r, RaycastHit &hit, bool clipBackfaces)const{
	const Ray inversedRay(r.origin, 1.0f / r.direction);
	const register TreeNode *root = (tree + 0);
	if (root->children == NULL) return castInLeaf(r, hit, 0, clipBackfaces);
	if (!Shapes::castPreInversed<AABB>(inversedRay, (tree + 0)->bounds, false)) return false;
	CastFrame stack[OCTREE_MAX_DEPTH + 1];
	int i = 0;
	configureCastFrame(stack[0], root->children, r);
	while (true){
		if (i < 0) return false;
		else{
			CastFrame &frame = stack[i];
			if (frame.curChild >= 8) i--;
			else {
				register char curChild = (frame.priorityChild ^ frame.curChild);
				const register TreeNode *child = frame.node + curChild;
				if (Shapes::castPreInversed<AABB>(inversedRay, child->bounds, false)) {
					const register TreeNode *children = child->children;
					if (children == NULL) {
						if (castInLeaf(r, hit, (int)(child - root), clipBackfaces)) return true;
					}
					else {
						i++;
						configureCastFrame(stack[i], children, r);
					}
				}
				frame.curChild++;
			}
		}
	}
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename ElemType>
// Returns current node count
__device__ __host__ inline int Octree<ElemType>::getNodeCount()const{
	return tree.size();
}
template<typename ElemType>
// "Dumps the internals" in the console (or whatever's mapped on the standard output)
__device__ __host__ inline void Octree<ElemType>::dump()const{
	printf("###################################\n");
	printf("TREE MASS: %d\n", tree.size());
	for (int i = 0; i < tree.size(); i++){
		printf("INDEX %d:\n    BOUNDS[(%.2f, %.2f, %,2f)-(%.2f, %.2f, %.2f)]\n", i, 
			tree[i].bounds.getMin().x, tree[i].bounds.getMin().y, tree[i].bounds.getMin().z, 
			tree[i].bounds.getMax().x, tree[i].bounds.getMax().y, tree[i].bounds.getMax().z);
		if (tree[i].children){
			const TreeNode *c = tree[i].children;
			const TreeNode *r = (tree + 0);
			printf("    CHILDREN: [%d, %d, %d, %d, %d, %d, %d, %d]\n", (c - r), (c + 1 - r), (c + 2 - r), (c + 3 - r), (c + 4 - r), (c + 5 - r), (c + 6 - r), (c + 7 - r));
		}
	}
	printf("\nNODE DATA MASS: %d\n", nodeData.size());
	for (int i = 0; i < nodeData.size(); i++){
		printf("INDEX: %d; MASS: %d\n    [", i, nodeData[i].size());
		for (int j = 0; j < nodeData[i].size(); j++){
			printf("%d", (nodeData[i][j] - (data + 0)));
			if (j < (nodeData[i].size() - 1)) printf(", ");
			else printf("]\n");
		}
	}
	printf("\nDATA MASS: %d\n", data.size());
	for (int i = 0; i < data.size(); i++){
		printf("INDEX: %d: ", i);
		Shapes::dump(data[i]);
		printf("\n");
	}
	printf("\n");
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename ElemType>
/* Uploads unit to CUDA device and returns the clone address */
inline Octree<ElemType>* Octree<ElemType>::upload()const{
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_BODY(Octree);
}
template<typename ElemType>
/* Uploads unit to the given location on the CUDA device (returns true, if successful; needs RAW data address) */
inline bool Octree<ElemType>::uploadAt(Octree *address)const{
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_AT_BODY(Octree);
}
template<typename ElemType>
/* Uploads given source array/unit to the given target location on CUDA device (returns true, if successful; needs RAW data address) */
inline bool Octree<ElemType>::upload(const Octree *source, Octree *target, int count){
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_AT_BODY(Octree);
}
template<typename ElemType>
/* Uploads given source array/unit to CUDA device and returns the clone address */
inline Octree<ElemType>* Octree<ElemType>::upload(const Octree<ElemType> *source, int count){
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_BODY(Octree);
}
template<typename ElemType>
/* Disposed given array/unit on CUDA device, making it ready to be free-ed (returns true, if successful) */
inline bool Octree<ElemType>::dispose(Octree *arr, int count){
	IMPLEMENT_CUDA_LOAD_INTERFACE_DISPOSE_BODY(Octree);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/


/** ========================================================== **/
template<typename ElemType>
__device__ __host__ inline Octree<ElemType>::TreeNode::TreeNode(AABB boundingBox){
	bounds = boundingBox;
	children = NULL;
}


/** ========================================================== **/
template<typename ElemType>
// Default constructor (does nothing)
__device__ __host__ inline Octree<ElemType>::RaycastHit::RaycastHit(){ }
template<typename ElemType>
// Constructs RaycastHit from the given parameters
__device__ __host__ inline Octree<ElemType>::RaycastHit::RaycastHit(const ElemType &elem, const float d, const Vector3 &p){
	set(elem, d, p);
}
template<typename ElemType>
// Constructs RaycastHit from the given parameters
__device__ __host__ inline Octree<ElemType>::RaycastHit& Octree<ElemType>::RaycastHit::operator()(const ElemType &elem, const float d, const Vector3 &p){
	set(elem, d, p);
	return (*this);
}
template<typename ElemType>
// Constructs RaycastHit from the given parameters
__device__ __host__ inline void Octree<ElemType>::RaycastHit::set(const ElemType &elem, const float d, const Vector3 &p){
	object = elem;
	hitDistance = d;
	hitPoint = p;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

/** ========================================================== **/
template<typename ElemType>
__device__ __host__ inline void Octree<ElemType>::fixTreeNodePointers(const TreeNode *falseRoot) {
	TreeNode *newRoot = (tree + 0);
	if (falseRoot != newRoot) {
		for (int i = 0; i < tree.size(); i++)
			if (tree[i].children != NULL)
				tree[i].children = (newRoot + (tree[i].children - falseRoot));
	}
}
template<typename ElemType>
__device__ __host__ inline void Octree<ElemType>::fixNodeDataPointers(const ElemType *falseRoot) {
	ElemType *newRoot = (data + 0);
	if (falseRoot != newRoot) {
		for (int i = 0; i < nodeData.size(); i++)
			for (int j = 0; j < nodeData[i].size(); j++)
				if (nodeData[i][j] != NULL) nodeData[i][j] = (newRoot + (nodeData[i][j] - falseRoot));
	}
}

/** ========================================================== **/
template<typename ElemType>
__device__ __host__ inline void Octree<ElemType>::pushData(const ElemType &object) {
	ElemType *oldRoot = (data + 0);
	data.push(object);
	fixNodeDataPointers(oldRoot);
}
template<typename ElemType>
__device__ __host__ inline void Octree<ElemType>::flushTree(){
	TreeNode *oldRoot = (tree + 0);
	tree.flush(8);
	nodeData.flush(8);
	fixTreeNodePointers(oldRoot);
}

/** ========================================================== **/
template<typename ElemType>
__device__ __host__ __noinline__ void Octree<ElemType>::split(int index, int depth){
	if (depth >= OCTREE_MAX_DEPTH) return;
	int nodeCount = nodeData[index].size();
	if (nodeCount < OCTREE_POLYCOUNT_TO_SPLIT_NODE) return;
	if (!splittingMakesSence(index)) return;

	Vector3 center(0, 0, 0);
	for (int i = 0; i < nodeData[index].size(); i++)
		center += Shapes::massCenter<ElemType>(*nodeData[index][i]);
	center /= ((float)nodeData[index].size());
	const Vertex bndStart = tree[index].bounds.getMin() + OCTREE_AABB_EPSILON_VECTOR;
	const Vertex bndEnd = tree[index].bounds.getMax() - OCTREE_AABB_EPSILON_VECTOR;
	if ((center.x < bndStart.x || center.y < bndStart.y || center.z < bndStart.z) || (center.x > bndEnd.x || center.y > bndEnd.y || center.z > bndEnd.z))
		center = (bndStart + bndEnd) / 2;
	splitNode(index, center);
	for (int i = 0; i < nodeData[index].size(); i++){
		const ElemType* dataPtr = nodeData[index][i];
		for (int j = 0; j < 8; j++){
			int childIndex = (int)(tree[index].children + j - (tree + 0));
			if (Shapes::intersect<AABB, ElemType>(tree[childIndex].bounds, *dataPtr))
				nodeData[childIndex].push(dataPtr);
		}
	}
	nodeData[index].clear();
	for (int i = 0; i < 8; i++)
		split((int)(tree[index].children + i - (tree + 0)), depth + 1);
}
template<typename ElemType>
__device__ __host__ inline bool Octree<ElemType>::splittingMakesSence(int index){
	const AABB &boundingBox = tree[index].bounds;
	const int nodeCount = (nodeData[index].size() - 1);
	const ElemType &elem = (*nodeData[index].top());
	for (int i = 0; i < nodeCount; i++)
		if (!Shapes::sharePoint<ElemType, AABB>(elem, *nodeData[index][i], boundingBox))
			return true;
	return false;
}
template<typename ElemType>
__device__ __host__ inline void Octree<ElemType>::splitNode(int index, Vertex center){
	int startChildId = tree.size();
	flushTree();
	TreeNode *startChild = (tree + startChildId);
	const Vertex start = tree[index].bounds.getMin();
	const Vertex end = tree[index].bounds.getMax();
	startChild->bounds = AABB(start, center + OCTREE_AABB_EPSILON_VECTOR);
	(startChild + 1)->bounds = AABB(Vertex(start.x, start.y, center.z - OCTREE_AABB_EPSILON), Vertex(center.x + OCTREE_AABB_EPSILON, center.y + OCTREE_AABB_EPSILON, end.z));
	(startChild + 2)->bounds = AABB(Vertex(start.x, center.y - OCTREE_AABB_EPSILON, start.z), Vertex(center.x + OCTREE_AABB_EPSILON, end.y, center.z + OCTREE_AABB_EPSILON));
	(startChild + 3)->bounds = AABB(Vertex(start.x, center.y - OCTREE_AABB_EPSILON, center.z - OCTREE_AABB_EPSILON), Vertex(center.x + OCTREE_AABB_EPSILON, end.y, end.z));
	(startChild + 4)->bounds = AABB(Vertex(center.x - OCTREE_AABB_EPSILON, start.y, start.z), Vertex(end.x, center.y + OCTREE_AABB_EPSILON, center.z + OCTREE_AABB_EPSILON));
	(startChild + 5)->bounds = AABB(Vertex(center.x - OCTREE_AABB_EPSILON, start.y, center.z - OCTREE_AABB_EPSILON), Vertex(end.x, center.y + OCTREE_AABB_EPSILON, end.z));
	(startChild + 6)->bounds = AABB(Vertex(center.x - OCTREE_AABB_EPSILON, center.y - OCTREE_AABB_EPSILON, start.z), Vertex(end.x, end.y, center.z + OCTREE_AABB_EPSILON));
	(startChild + 7)->bounds = AABB(center - OCTREE_AABB_EPSILON_VECTOR, end);
	tree[index].children = startChild;
	for (int i = 0; i < 8; i++){
		nodeData[(int)((startChild + i) - (tree + 0))].clear();
	}
}
template<typename ElemType>
__device__ __host__ inline void Octree<ElemType>::reduceNode(int index){
	Vertex start = tree[index].bounds.getMax();
	Vertex end = tree[index].bounds.getMin();
	if (tree[index].children != NULL) {
		for (int i = 0; i < 8; i++)
			reduceNode((int)((tree[index].children + i) - (tree + 0)));
		for (int i = 0; i < 8; i++) {
			AABB bounds = tree[index].children[i].bounds;
			const Vertex bStart = bounds.getMin();
			const Vertex bEnd = bounds.getMax();
			start(min(start.x, bStart.x), min(start.y, bStart.y), min(start.z, bStart.z));
			end(max(end.x, bEnd.x), max(end.y, bEnd.y), max(end.z, bEnd.z));
		}
	}
	else for (int i = 0; i < nodeData[index].size(); i++) {
		AABB objectBounds = Shapes::boundingBox<ElemType>(*nodeData[index][i]);
		const Vertex bStart = objectBounds.getMin();
		const Vertex bEnd = objectBounds.getMax();
		start(min(start.x, bStart.x), min(start.y, bStart.y), min(start.z, bStart.z));
		end(max(end.x, bEnd.x), max(end.y, bEnd.y), max(end.z, bEnd.z));
	}
	start -= OCTREE_AABB_EPSILON_VECTOR;
	end += OCTREE_AABB_EPSILON_VECTOR;
	const Vertex bStart = tree[index].bounds.getMin();
	const Vertex bEnd = tree[index].bounds.getMax();
	start(max(start.x, bStart.x), max(start.y, bStart.y), max(start.z, bStart.z));
	end(min(end.x, bEnd.x), min(end.y, bEnd.y), min(end.z, bEnd.z));
	tree[index].bounds = AABB(start, end);
}


/** ========================================================== **/
template<typename ElemType>
__device__ __host__ __noinline__ void Octree<ElemType>::put(const ElemType *elem, int nodeIndex, int depth){
	if (Shapes::intersect<AABB, ElemType>(tree[nodeIndex].bounds, *elem)){
		if (tree[nodeIndex].children == NULL){
			nodeData[nodeIndex].push(elem);
			if (depth >= OCTREE_MAX_DEPTH) return;

			int nodeCount = nodeData[nodeIndex].size();
			if (nodeCount < OCTREE_POLYCOUNT_TO_SPLIT_NODE) return;

			if (splittingMakesSence(nodeIndex)){
				splitNode(nodeIndex, tree[nodeIndex].bounds.getCenter());
				for (int i = 0; i < nodeData[nodeIndex].size(); i++){
					for (int j = 0; j < 8; j++) put(nodeData[nodeIndex][i], (int)(tree[nodeIndex].children + j - (tree + 0)), depth + 1);
				}
				nodeData[nodeIndex].clear();
			}
		}
		else{
			for (int i = 0; i < 8; i++){
				int child = (int)(tree[nodeIndex].children + i - (tree + 0));
				put(elem, child, depth + 1);
			}
		}
	}
}


/** ========================================================== **/
template<typename ElemType>
__device__ __host__ inline void Octree<ElemType>::configureCastFrame(CastFrame &frame, const TreeNode *children, const Ray &r) {
	frame.node = children;
	if (r.direction.z < 0) frame.priorityChild = 1;
	else frame.priorityChild = 0;
	if (r.direction.y < 0) frame.priorityChild += 2;
	if (r.direction.x < 0) frame.priorityChild += 4;
	frame.curChild = 0;
}
template<typename ElemType>
__device__ __host__ inline bool Octree<ElemType>::castInLeaf(const Ray &r, RaycastHit &hit, int index, bool clipBackfaces)const{
	const Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> &nodeTris = nodeData[index];
	if (nodeTris.size() <= 0) return false;
	const AABB &bounds = tree[index].bounds;

	float bestDistance = FLT_MAX;
	Vertex bestHitPoint;
	const ElemType *bestHit = NULL;
	for (int i = 0; i < nodeTris.size(); i++){
		const ElemType *object = nodeTris[i];
		Vertex hitPoint;
		float distance;
		bool casted = Shapes::cast<ElemType>(r, *object, distance, hitPoint, clipBackfaces);
		if (casted && distance < bestDistance && bounds.contains(hitPoint)){
			bestDistance = distance;
			bestHitPoint = hitPoint;
			bestHit = object;
		}
	}
	if (bestDistance != FLT_MAX){
		hit.set(*bestHit, bestDistance, bestHitPoint);
		return true;
	}
	else return false;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Private kernels: **/

namespace OctreePrivateKernels{
#define OCTREE_PRIVATE_KERNELS_THREADS_PER_BLOCK 256
#define OCTREE_PRIVATE_KERNELS_UNITS_PER_THREAD 16

	__device__ __host__ inline static int numThreads(){
		return OCTREE_PRIVATE_KERNELS_THREADS_PER_BLOCK;
	}
	__device__ __host__ inline static int unitsPerThread(){
		return OCTREE_PRIVATE_KERNELS_UNITS_PER_THREAD;
	}
	__device__ __host__ inline static int unitsPerBlock(){
		return (OCTREE_PRIVATE_KERNELS_THREADS_PER_BLOCK * OCTREE_PRIVATE_KERNELS_UNITS_PER_THREAD);
	}
	__device__ __host__ inline static int numBlocks(int totalNumber){
		int perBlock = unitsPerBlock();
		return ((totalNumber + perBlock - 1) / perBlock);
	}

#undef OCTREE_PRIVATE_KERNELS_THREADS_PER_BLOCK
#undef OCTREE_PRIVATE_KERNELS_UNITS_PER_THREAD

	__global__ static void fixRoots(const Octree<>::TreeNode *hostRoot, Octree<>::TreeNode *devRoot, int count){
		int start = (blockIdx.x * unitsPerBlock() + threadIdx.x * unitsPerThread());
		int end = start + unitsPerThread();
		if (end > count) end = count;
		Octree<>::TreeNode *endNode = (devRoot + end);
		for (Octree<>::TreeNode *ptr = (devRoot + start); ptr < endNode; ptr++)
			if (ptr->children != NULL) ptr->children = (devRoot + (ptr->children - hostRoot));
	}
	template<typename ElemType>
	__global__ static void fixNodeDataPointers(const ElemType *hostRoot, const ElemType *devRoot, Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> *pointers, int count) {
		int start = (blockIdx.x * unitsPerBlock() + threadIdx.x * unitsPerThread());
		int end = start + unitsPerThread();
		if (end > count) end = count;
		Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> *endNode = (pointers + end);
		for (Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> *ptr = (pointers + start); ptr < endNode; ptr++)
			for (int i = 0; i < ptr->size(); i++) if ((*ptr)[i] != NULL) (*ptr)[i] = (devRoot + ((*ptr)[i] - hostRoot));
	}
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/

template<typename ElemType>
__device__ __host__ inline void TypeTools<Octree<ElemType> >::init(Octree<ElemType> &m){
	TypeTools<Stacktor<Octree<>::TreeNode> >::init(m.tree);
	TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::init(m.nodeData);
	TypeTools<Stacktor<ElemType > >::init(m.data);
}
template<typename ElemType>
__device__ __host__ inline void TypeTools<Octree<ElemType> >::dispose(Octree<ElemType> &m){
	TypeTools<Stacktor<Octree<>::TreeNode> >::dispose(m.tree);
	TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::dispose(m.nodeData);
	TypeTools<Stacktor<ElemType > >::dispose(m.data);
}
template<typename ElemType>
__device__ __host__ inline void TypeTools<Octree<ElemType> >::swap(Octree<ElemType> &a, Octree<ElemType> &b){
	TypeTools<Stacktor<Octree<>::TreeNode> >::swap(a.tree, b.tree);
	TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::swap(a.nodeData, b.nodeData);
	TypeTools<Stacktor<ElemType > >::swap(a.data, b.data);
}
template<typename ElemType>
__device__ __host__ inline void TypeTools<Octree<ElemType> >::transfer(Octree<ElemType> &src, Octree<ElemType> &dst){
	TypeTools<Stacktor<Octree<>::TreeNode> >::transfer(src.tree, dst.tree);
	TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::transfer(src.nodeData, dst.nodeData);
	TypeTools<Stacktor<ElemType > >::transfer(src.data, dst.data);
}

template<typename ElemType>
inline bool TypeTools<Octree<ElemType> >::prepareForCpyLoad(const Octree<ElemType> *source, Octree<ElemType> *hosClone, Octree<ElemType> *devTarget, int count){
	int i = 0;
	for (i = 0; i < count; i++){
		if (!TypeTools<Stacktor<Octree<>::TreeNode> >::prepareForCpyLoad(&source[i].tree, &hosClone[i].tree, &((devTarget + i)->tree), 1)) break;
		if (!TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::prepareForCpyLoad(&source[i].nodeData, &hosClone[i].nodeData, &((devTarget + i)->nodeData), 1)){
			TypeTools<Stacktor<Octree<>::TreeNode> >::undoCpyLoadPreparations(&source[i].tree, &hosClone[i].tree, &((devTarget + i)->tree), 1);
			break;
		}
		if (!TypeTools<Stacktor<ElemType > >::prepareForCpyLoad(&source[i].data, &hosClone[i].data, &((devTarget + i)->data), 1)){
			TypeTools<Stacktor<Octree<>::TreeNode> >::undoCpyLoadPreparations(&source[i].tree, &hosClone[i].tree, &((devTarget + i)->tree), 1);
			TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::undoCpyLoadPreparations(&source[i].nodeData, &hosClone[i].nodeData, &((devTarget + i)->nodeData), 1);
			break;
		}
		bool streamError = false;
		cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) streamError = true;
		else {
			OctreePrivateKernels::fixRoots<<<OctreePrivateKernels::numBlocks(source[i].tree.size()), OctreePrivateKernels::numThreads(), 0, stream>>>(source[i].tree + 0, hosClone[i].tree + 0, source[i].tree.size());
			OctreePrivateKernels::fixNodeDataPointers<ElemType><<<OctreePrivateKernels::numBlocks(source[i].nodeData.size()), OctreePrivateKernels::numThreads(), 0, stream>>>(source[i].data + 0, hosClone[i].data + 0, hosClone[i].nodeData + 0, source[i].nodeData.size());
			streamError = (cudaStreamSynchronize(stream) != cudaSuccess);
			if (cudaStreamDestroy(stream) != cudaSuccess) streamError = true;
		}
		if (streamError) {
			TypeTools<Stacktor<Octree<>::TreeNode> >::undoCpyLoadPreparations(&source[i].tree, &hosClone[i].tree, &((devTarget + i)->tree), 1);
			TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::undoCpyLoadPreparations(&source[i].nodeData, &hosClone[i].nodeData, &((devTarget + i)->nodeData), 1);
			TypeTools<Stacktor<ElemType > >::undoCpyLoadPreparations(&source[i].data, &hosClone[i].data, &((devTarget + i)->data), 1);
			break;
		}
	}
	if (i < count){
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return(false);
	}
	return(true);
}

template<typename ElemType>
inline void TypeTools<Octree<ElemType> >::undoCpyLoadPreparations(const Octree<ElemType> *source, Octree<ElemType> *hosClone, Octree<ElemType> *devTarget, int count){
	for (int i = 0; i < count; i++){
		TypeTools<Stacktor<Octree<>::TreeNode> >::undoCpyLoadPreparations(&source[i].tree, &hosClone[i].tree, &((devTarget + i)->tree), 1);
		TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::undoCpyLoadPreparations(&source[i].nodeData, &hosClone[i].nodeData, &((devTarget + i)->nodeData), 1);
		TypeTools<Stacktor<ElemType > >::undoCpyLoadPreparations(&source[i].data, &hosClone[i].data, &((devTarget + i)->data), 1);
	}
}
template<typename ElemType>
inline bool TypeTools<Octree<ElemType> >::devArrayNeedsToBeDisoposed(){
	return(TypeTools<Octree<>::TreeNode>::devArrayNeedsToBeDisoposed() || TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::devArrayNeedsToBeDisoposed() || TypeTools<ElemType >::devArrayNeedsToBeDisoposed());
}
template<typename ElemType>
inline bool TypeTools<Octree<ElemType> >::disposeDevArray(Octree<ElemType> *arr, int count){
	for (int i = 0; i < count; i++){
		if (!TypeTools<Stacktor<Octree<>::TreeNode> >::disposeDevArray(&((arr + i)->tree), 1)) return false;
		if (!TypeTools<Stacktor<Stacktor<const ElemType*, OCTREE_VOXEL_LOCAL_CAPACITY> > >::disposeDevArray(&((arr + i)->nodeData), 1)) return false;
		if (!TypeTools<Stacktor<ElemType > >::disposeDevArray(&((arr + i)->data), 1)) return false;
	}
	return(true);
}





#undef OCTREE_AABB_EPSILON_MULTIPLIER
#undef OCTREE_AABB_EPSILON
#undef OCTREE_AABB_EPSILON_VECTOR


