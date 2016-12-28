#include"Camera.cuh"





__dumb__ Photon Camera::getPhoton(const Vector2 &screenSpacePosition)const {
	return (lense.getScreenPhoton(screenSpacePosition) >> transform);
}








/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<>
__device__ __host__ inline void TypeTools<Camera>::init(Camera &m) {
	TypeTools<Lense>::init(m.lense);
	TypeTools<Transform>::init(m.transform);
}
template<>
__device__ __host__ inline void TypeTools<Camera>::dispose(Camera &m) {
	TypeTools<Lense>::dispose(m.lense);
	TypeTools<Transform>::dispose(m.transform);
}
template<>
__device__ __host__ inline void TypeTools<Camera>::swap(Camera &a, Camera &b) {
	TypeTools<Lense>::swap(a.lense, b.lense);
	TypeTools<Transform>::swap(a.transform, b.transform);
}
template<>
__device__ __host__ inline void TypeTools<Camera>::transfer(Camera &src, Camera &dst) {
	TypeTools<Lense>::transfer(src.lense, dst.lense);
	TypeTools<Transform>::transfer(src.transform, dst.transform);
}

template<>
inline bool TypeTools<Camera>::prepareForCpyLoad(const Camera *source, Camera *hosClone, Camera *devTarget, int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if (!TypeTools<Lense>::prepareForCpyLoad(&(source + i)->lense, &(hosClone + i)->lense, &(devTarget + i)->lense, 1)) break;
		if (!TypeTools<Transform>::prepareForCpyLoad(&(source + i)->transform, &(hosClone + i)->transform, &(devTarget + i)->transform, 1)) {
			TypeTools<Lense>::undoCpyLoadPreparations(&(source + i)->lense, &(hosClone + i)->lense, &(devTarget + i)->lense, 1);
			break;
		}
	}
	if (i < count) {
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return(false);
	}
	return(true);
}

template<>
inline void TypeTools<Camera>::undoCpyLoadPreparations(const Camera *source, Camera *hosClone, Camera *devTarget, int count) {
	for (int i = 0; i < count; i++) {
		TypeTools<Lense>::undoCpyLoadPreparations(&(source + i)->lense, &(hosClone + i)->lense, &(devTarget + i)->lense, 1);
		TypeTools<Transform>::undoCpyLoadPreparations(&(source + i)->transform, &(hosClone + i)->transform, &(devTarget + i)->transform, 1);
	}
}

template<>
inline bool TypeTools<Camera>::devArrayNeedsToBeDisposed() {
	return (TypeTools<Lense>::devArrayNeedsToBeDisposed() || TypeTools<Transform>::devArrayNeedsToBeDisposed());
}
template<>
inline bool TypeTools<Camera>::disposeDevArray(Camera *arr, int count) {
	for (int i = 0; i < count; i++) {
		if (!TypeTools<Lense>::disposeDevArray(&(arr + i)->lense, 1)) return(false);
		if (!TypeTools<Transform>::disposeDevArray(&(arr + i)->transform, 1)) return(false);
	}
	return(true);
}
