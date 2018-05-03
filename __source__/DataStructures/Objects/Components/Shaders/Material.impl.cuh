#include"Material.cuh"
#include"../../../../Namespaces/Shapes/Shapes.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
__dumb__ void Shader<HitType>::clean() {
	//castFunction = NULL;
	bounceFunction = NULL;
	illuminateFunction = NULL;

	requestIndirectSamplesFn = NULL;
	getReflectedColorFn = NULL;
}
template<typename HitType>
template<typename ShaderType>
__dumb__ void Shader<HitType>::use() {
	//castFunction = castGeneric<ShaderType>;
	bounceFunction = bounceGeneric<ShaderType>;
	illuminateFunction = illuminateGeneric<ShaderType>;

	requestIndirectSamplesFn = requestIndirectSamplesGeneric<ShaderType>;
	getReflectedColorFn = getReflectedColorGeneric<ShaderType>;
}
/*
template<typename HitType>
__dumb__ ShaderReport Shader<HitType>::cast(const void *shader, const ShaderHitInfo<HitType>& info)const {
	return castFunction(shader, info);
}
*/
template<typename HitType>
__dumb__ void Shader<HitType>::bounce(const void *shader, const ShaderBounceInfo<HitType> &info, PhotonPack &result)const {
	return bounceFunction(shader, info, result);
}
template<typename HitType>
__dumb__ Photon Shader<HitType>::illuminate(const void *shader, const ShaderHitInfo<HitType>& info)const {
	return illuminateFunction(shader, info);
}

template<typename HitType>
__dumb__ void Shader<HitType>::requestIndirectSamples(const void *shader, const ShaderInirectSamplesRequest<HitType> &request, RaySamples *samples)const {
	requestIndirectSamplesFn(shader, request, samples);
}
template<typename HitType>
__dumb__ Color Shader<HitType>::getReflectedColor(const void *shader, const ShaderReflectedColorRequest<HitType> &request)const {
	return getReflectedColorFn(shader, request);
}


/*
template<typename HitType>
template<typename ShaderType>
__dumb__ ShaderReport Shader<HitType>::castGeneric(const void *shader, const ShaderHitInfo<HitType>& info) {
	return ((ShaderType*)shader)->cast(info);
}
*/
template<typename HitType>
template<typename ShaderType>
__dumb__  void Shader<HitType>::bounceGeneric(const void *shader, const ShaderBounceInfo<HitType> &info, PhotonPack &result) {
	return ((ShaderType*)shader)->bounce(info, result);
}
template<typename HitType>
template<typename ShaderType>
__dumb__ Photon Shader<HitType>::illuminateGeneric(const void *shader, const ShaderHitInfo<HitType>& info)  {
	return ((ShaderType*)shader)->illuminate(info);
}

template<typename HitType>
template<typename ShaderType>
__dumb__ void Shader<HitType>::requestIndirectSamplesGeneric(const void *shader, const ShaderInirectSamplesRequest<HitType> &request, RaySamples *samples) {
	return ((ShaderType*)shader)->requestIndirectSamples(request, samples);
}
template<typename HitType>
template<typename ShaderType>
__dumb__ Color Shader<HitType>::getReflectedColorGeneric(const void *shader, const ShaderReflectedColorRequest<HitType> &request) {
	return ((ShaderType*)shader)->getReflectedColor(request);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/*
template<typename HitType>
__dumb__ ShaderReport Material<HitType>::cast(const ShaderHitInfo<HitType>& info)const {
	return Generic<Shader<HitType> >::functions().cast(Generic<Shader<HitType> >::object(), info);
}
*/
template<typename HitType>
__dumb__ void Material<HitType>::bounce(const ShaderBounceInfo<HitType> &info, PhotonPack &result)const {
	return Generic<Shader<HitType> >::functions().bounce(Generic<Shader<HitType> >::object(), info, result);
}
template<typename HitType>
__dumb__ Photon Material<HitType>::illuminate(const ShaderHitInfo<HitType>& info)const {
	return Generic<Shader<HitType> >::functions().illuminate(Generic<Shader<HitType> >::object(), info);
}

template<typename HitType>
__dumb__ void Material<HitType>::requestIndirectSamples(const ShaderInirectSamplesRequest<HitType> &request, RaySamples *samples)const {
	Generic<Shader<HitType> >::functions().requestIndirectSamples(Generic<Shader<HitType> >::object(), request, samples);
}
template<typename HitType>
__dumb__ Color Material<HitType>::getReflectedColor(const ShaderReflectedColorRequest<HitType> &request)const {
	return Generic<Shader<HitType> >::functions().getReflectedColor(Generic<Shader<HitType> >::object(), request);
}


template<typename HitType>
inline Material<HitType>* Material<HitType>::upload()const {
	return ((Material*)Generic<Shader<HitType> >::upload());
}
template<typename HitType>
inline Material<HitType>* Material<HitType>::upload(const Material *source, int count) {
	return ((Material*)Generic<Shader<HitType> >::upload(source, count));
}






/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<typename HitType>
__device__ __host__ inline void TypeTools<Material<HitType> >::init(Material<HitType> &m) {
	TypeTools<Generic<Shader<HitType> > >::init(m);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Material<HitType> >::dispose(Material<HitType> &m) {
	TypeTools<Generic<Shader<HitType> > >::dispose(m);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Material<HitType> >::swap(Material<HitType> &a, Material<HitType> &b) {
	TypeTools<Generic<Shader<HitType> > >::swap(a, b);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Material<HitType> >::transfer(Material<HitType> &src, Material<HitType> &dst) {
	TypeTools<Generic<Shader<HitType> > >::transfer(src, dst);
}

template<typename HitType>
inline bool TypeTools<Material<HitType> >::prepareForCpyLoad(const Material<HitType> *source, Material<HitType> *hosClone, Material<HitType> *devTarget, int count) {
	return TypeTools<Generic<Shader<HitType> > >::prepareForCpyLoad(source, hosClone, devTarget, count);
}

template<typename HitType>
inline void TypeTools<Material<HitType> >::undoCpyLoadPreparations(const Material<HitType> *source, Material<HitType> *hosClone, Material<HitType> *devTarget, int count) {
	TypeTools<Generic<Shader<HitType> > >::undoCpyLoadPreparations(source, hosClone, devTarget, count);
}

template<typename HitType>
inline bool TypeTools<Material<HitType> >::devArrayNeedsToBeDisposed() {
	return TypeTools<Generic<Shader<HitType> > >::devArrayNeedsToBeDisposed();
}
template<typename HitType>
inline bool TypeTools<Material<HitType> >::disposeDevArray(Material<HitType> *arr, int count) {
	return TypeTools<Generic<Shader<HitType> > >::disposeDevArray(arr, count);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
__dumb__ Renderable<HitType>::Renderable() { }
template<typename HitType>
__dumb__ Renderable<HitType>::Renderable(const HitType &obj, int matId) {
	object = obj; materialId = matId; 
}
template<typename HitType>
__dumb__ bool Renderable<HitType>::intersects(const Renderable &other)const {
	return Shapes::intersect<HitType>(object, other.object);
}
template<typename HitType>
__dumb__ bool Renderable<HitType>::intersects(const AABB &other)const {
	return Shapes::intersect<AABB, HitType>(other, object);
}
template<typename HitType>
__dumb__ bool Renderable<HitType>::cast(const Ray& r, bool clipBackface)const {
	return Shapes::cast<HitType>(r, object, clipBackface);
}
template<typename HitType>
__dumb__ bool Renderable<HitType>::castPreInversed(const Ray& inversedRay, bool clipBackface)const {
	return Shapes::castPreInversed<HitType>(inversedRay, object, clipBackface);
}
template<typename HitType>
__dumb__ bool Renderable<HitType>::cast(const Ray& ray, float &hitDistance, Vertex& hitPoint, bool clipBackface)const {
	return Shapes::cast<HitType>(ray, object, hitDistance, hitPoint, clipBackface);
}
template<typename HitType>
template<typename BoundType>
__dumb__ bool Renderable<HitType>::sharesPoint(const Renderable& b, const BoundType& commonPointBounds)const {
	return Shapes::sharePoint<HitType, BoundType>(object, b.object, commonPointBounds);
}
template<typename HitType>
template<typename Shape>
__dumb__ Vertex Renderable<HitType>::intersectionCenter(const Shape &shape)const {
	return Shapes::intersectionCenter<Shape, HitType>(shape, object);
}
template<typename HitType>
template<typename Shape>
__dumb__ AABB Renderable<HitType>::intersectionBounds(const Shape &shape)const {
	return Shapes::intersectionBounds<Shape, HitType>(shape, object);
}
template<typename HitType>
__dumb__ Vertex Renderable<HitType>::massCenter()const {
	return Shapes::massCenter<HitType>(object);
}
template<typename HitType>
__dumb__ AABB Renderable<HitType>::boundingBox()const {
	return Shapes::boundingBox<HitType>(object);
}
template<typename HitType>
__dumb__ void Renderable<HitType>::dump()const {
	printf("Renderable Object: {\n");
	printf("Object: \n");
	Shapes::dump<HitType>(object);
	printf("\nMaterialId: %d}\n", materialId);
	printf("\n}\n");
}


template<typename HitType>
__device__ __host__ inline void TypeTools<Renderable<HitType> >::init(Renderable<HitType> &m) { TypeTools<HitType>::init(m.object); m.materialId = -1; }
template<typename HitType>
__device__ __host__ inline void TypeTools<Renderable<HitType> >::dispose(Renderable<HitType> &m) { TypeTools<HitType>::dispose(m.object); m.materialId = -1; }
template<typename HitType>
__device__ __host__ inline void TypeTools<Renderable<HitType> >::swap(Renderable<HitType> &a, Renderable<HitType> &b) {
	TypeTools<HitType>::swap(a.object, b.object);
	TypeTools<int>::swap(a.materialId, b.materialId);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Renderable<HitType> >::transfer(Renderable<HitType> &src, Renderable<HitType> &dst) {
	TypeTools<HitType>::transfer(src.object, dst.object);
	TypeTools<int>::transfer(src.materialId, dst.materialId);
}



template<typename HitType>
inline bool TypeTools<Renderable<HitType> >::prepareForCpyLoad(const Renderable<HitType> *source, Renderable<HitType> *hosClone, Renderable<HitType> *devTarget, int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if (!TypeTools<HitType>::prepareForCpyLoad(&source[i].object, &hosClone[i].object, &((devTarget + i)->object), 1)) break;
		hosClone[i].materialId = source[i].materialId;
	}
	if (i < count) { 
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return false;
	}
	return true;
}

template<typename HitType>
inline void TypeTools<Renderable<HitType> >::undoCpyLoadPreparations(const Renderable<HitType> *source, Renderable<HitType> *hosClone, Renderable<HitType> *devTarget, int count) {
	for (int i = 0; i < count; i++) TypeTools<HitType>::undoCpyLoadPreparations(&source[i].object, &hosClone[i].object, &((devTarget + i)->object), 1);
}
template<typename HitType>
inline bool TypeTools<Renderable<HitType> >::devArrayNeedsToBeDisposed() {
	return TypeTools<HitType>::devArrayNeedsToBeDisposed();
}
template<typename HitType>
inline bool TypeTools<Renderable<HitType> >::disposeDevArray(Renderable<HitType> *arr, int count) {
	for (int i = 0; i < count; i++) if (!TypeTools<HitType>::disposeDevArray(&((arr + i)->object), 1)) return false;
	return true;
}
