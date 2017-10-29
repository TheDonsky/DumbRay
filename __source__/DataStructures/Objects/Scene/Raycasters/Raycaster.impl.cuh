#include"Raycaster.cuh"



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename ElemType>
// Default constructor (does nothing)
__device__ __host__ inline RaycastHit<ElemType>::RaycastHit() { }
template<typename ElemType>
// Constructs RaycastHit from the given parameters
__device__ __host__ inline RaycastHit<ElemType>::RaycastHit(const ElemType &elem, const float d, const Vector3 &p) {
	set(elem, d, p);
}
template<typename ElemType>
// Constructs RaycastHit from the given parameters
__device__ __host__ inline RaycastHit<ElemType>& RaycastHit<ElemType>::operator()(const ElemType &elem, const float d, const Vector3 &p) {
	set(elem, d, p);
	return (*this);
}
template<typename ElemType>
// Constructs RaycastHit from the given parameters
__device__ __host__ inline void RaycastHit<ElemType>::set(const ElemType &elem, const float d, const Vector3 &p) {
	object = &elem;
	hitDistance = d;
	hitPoint = p;
}



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

template<typename HitType>
__dumb__ RaycastFunctionPack<HitType>::RaycastFunctionPack() {
	clean();
}

template<typename HitType>
__dumb__ void RaycastFunctionPack<HitType>::clean() {
	castFunction = NULL;
}
template<typename HitType>
template<typename RaycasterType>
__dumb__ void RaycastFunctionPack<HitType>::use() {
	castFunction = castGeneric<RaycasterType>;
}

template<typename HitType>
// Casts a ray (returns true if the ray hits something; result is written in hit)
__dumb__ bool RaycastFunctionPack<HitType>::cast(const void *raycaster, const Ray &r, RaycastHit<HitType> &hit, bool clipBackfaces, CastBreaker castBreaker)const {
	return castFunction(raycaster, r, hit, clipBackfaces, castBreaker);
}

template<typename HitType>
template<typename RaycasterType>
__dumb__ bool RaycastFunctionPack<HitType>::castGeneric(const void *raycaster, const Ray &r, RaycastHit<HitType> &hit, bool clipBackfaces, CastBreaker castBreaker) {
	return ((const RaycasterType*)raycaster)->cast(r, hit, clipBackfaces, castBreaker);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
__dumb__ bool Raycaster<HitType>::cast(const Ray &r, RaycastHit<HitType> &hit, bool clipBackfaces, CastBreaker castBreaker)const {
	return Generic<RaycastFunctionPack<HitType> >::functions().cast(Generic<RaycastFunctionPack<HitType> >::object(), r, hit, clipBackfaces, castBreaker);
}

template<typename HitType>
inline Raycaster<HitType>* Raycaster<HitType>::upload()const {
	return ((Raycaster*)Generic<RaycastFunctionPack<HitType> >::upload());
}
template<typename HitType>
inline Raycaster<HitType>* Raycaster<HitType>::upload(const Raycaster *source, int count) {
	return ((Raycaster*)Generic<RaycastFunctionPack<HitType> >::upload(source, count));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<typename HitType>
__device__ __host__ inline void TypeTools<Raycaster<HitType> >::init(Raycaster<HitType> &m) {
	TypeTools<Generic<RaycastFunctionPack<HitType> > >::init(m);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Raycaster<HitType> >::dispose(Raycaster<HitType> &m) {
	TypeTools<Generic<RaycastFunctionPack<HitType> > >::dispose(m);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Raycaster<HitType> >::swap(Raycaster<HitType> &a, Raycaster<HitType> &b) {
	TypeTools<Generic<RaycastFunctionPack<HitType> > >::swap(a, b);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Raycaster<HitType> >::transfer(Raycaster<HitType> &src, Raycaster<HitType> &dst) {
	TypeTools<Generic<RaycastFunctionPack<HitType> > >::transfer(src, dst);
}

template<typename HitType>
inline bool TypeTools<Raycaster<HitType> >::prepareForCpyLoad(const Raycaster<HitType> *source, Raycaster<HitType> *hosClone, Raycaster<HitType> *devTarget, int count) {
	return TypeTools<Generic<RaycastFunctionPack<HitType> > >::prepareForCpyLoad(source, hosClone, devTarget, count);
}

template<typename HitType>
inline void TypeTools<Raycaster<HitType> >::undoCpyLoadPreparations(const Raycaster<HitType> *source, Raycaster<HitType> *hosClone, Raycaster<HitType> *devTarget, int count) {
	TypeTools<Generic<RaycastFunctionPack<HitType> > >::undoCpyLoadPreparations(source, hosClone, devTarget, count);
}

template<typename HitType>
inline bool TypeTools<Raycaster<HitType> >::devArrayNeedsToBeDisposed() {
	return TypeTools<Generic<RaycastFunctionPack<HitType> > >::devArrayNeedsToBeDisposed();
}
template<typename HitType>
inline bool TypeTools<Raycaster<HitType> >::disposeDevArray(Raycaster<HitType> *arr, int count) {
	return TypeTools<Generic<RaycastFunctionPack<HitType> > >::disposeDevArray(arr, count);
}


