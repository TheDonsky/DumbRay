#include"Light.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename IlluminatedType>
__dumb__ void LightInterface<IlluminatedType>::clean() {
}
template<typename IlluminatedType>
template<typename LightType>
__dumb__ void LightInterface<IlluminatedType>::use() {
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename IlluminatedType>
inline Light<IlluminatedType>* Light<IlluminatedType>::upload()const {
	return (Light*)(Generic<LightInterface<IlluminatedType> >::upload());
}
template<typename IlluminatedType>
inline Light<IlluminatedType>* Light<IlluminatedType>::upload(const Light *source, int count) {
	return (Light*)(Generic<LightInterface<IlluminatedType> >::upload(source, count));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<typename IlluminatedType>
__device__ __host__ inline void TypeTools<Light<IlluminatedType> >::init(Light<IlluminatedType> &m) {
	TypeTools<Generic<LightInterface<IlluminatedType> > >::init(m);
}
template<typename IlluminatedType>
__device__ __host__ inline void TypeTools<Light<IlluminatedType> >::dispose(Light<IlluminatedType> &m) {
	TypeTools<Generic<LightInterface<IlluminatedType> > >::dispose(m);
}
template<typename IlluminatedType>
__device__ __host__ inline void TypeTools<Light<IlluminatedType> >::swap(Light<IlluminatedType> &a, Light<IlluminatedType> &b) {
	TypeTools<Generic<LightInterface<IlluminatedType> > >::swap(a, b);
}
template<typename IlluminatedType>
__device__ __host__ inline void TypeTools<Light<IlluminatedType> >::transfer(Light<IlluminatedType> &src, Light<IlluminatedType> &dst) {
	TypeTools<Generic<LightInterface<IlluminatedType> > >::transfer(src, dst);
}

template<typename IlluminatedType>
inline bool TypeTools<Light<IlluminatedType> >::prepareForCpyLoad(const Light<IlluminatedType> *source, Light<IlluminatedType> *hosClone, Light<IlluminatedType> *devTarget, int count) {
	return TypeTools<Generic<LightInterface<IlluminatedType> > >::prepareForCpyLoad(source, hosClone, devTarget, count);
}

template<typename IlluminatedType>
inline void TypeTools<Light<IlluminatedType> >::undoCpyLoadPreparations(const Light<IlluminatedType> *source, Light<IlluminatedType> *hosClone, Light<IlluminatedType> *devTarget, int count) {
	TypeTools<Generic<LightInterface<IlluminatedType> > >::undoCpyLoadPreparations(source, hosClone, devTarget, count);
}

template<typename IlluminatedType>
inline bool TypeTools<Light<IlluminatedType> >::devArrayNeedsToBeDisoposed() {
	return TypeTools<Generic<LightInterface<IlluminatedType> > >::devArrayNeedsToBeDisposed();
}
template<typename IlluminatedType>
inline bool TypeTools<Light<IlluminatedType> >::disposeDevArray(Light<IlluminatedType> *arr, int count) {
	return TypeTools<Generic<LightInterface<IlluminatedType> > >::disposeDevArray(arr, count);
}




