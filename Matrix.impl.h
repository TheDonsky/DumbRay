#include"Matrix.h"




/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename Type>
__device__ __host__ inline Matrix<Type>::Matrix(){
	matWidth = 0;
	matHeight = 0;
}
template<typename Type>
__device__ __host__ inline Matrix<Type>::Matrix(int width, int height){
	setDimensions(width, height);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename Type>
__device__ __host__ inline Type* Matrix<Type>::operator[](int y){
	return (data + (y * matWidth));
}
template<typename Type>
__device__ __host__ inline const Type* Matrix<Type>::operator[](int y)const{
	return (data + (y * matWidth));
}
template<typename Type>
__device__ __host__ inline Type& Matrix<Type>::operator()(int y, int x){
	return data[(y * matWidth) + x];
}
template<typename Type>
__device__ __host__ inline const Type& Matrix<Type>::operator()(int y, int x)const{
	return data[(y * matWidth) + x];
}
template<typename Type>
__device__ __host__ inline int Matrix<Type>::width()const{
	return matWidth;
}
template<typename Type>
__device__ __host__ inline int Matrix<Type>::height()const{
	return matHeight;
}
template<typename Type>
__device__ __host__ inline int Matrix<Type>::surface()const{
	return matWidth * matHeight;
}
template<typename Type>
__device__ __host__ inline void Matrix<Type>::clear(){
	matWidth = 0;
	matHeight = 0;
	data.clear();
}
template<typename Type>
__device__ __host__ inline void Matrix<Type>::setDimensions(int width, int height){
	matWidth = width;
	matHeight = height;
	data.clear();
	data.flush(matWidth * matHeight);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename Type>
/* Uploads unit to CUDA device and returns the clone address */
inline Matrix<Type>* Matrix<Type>::upload()const {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_BODY(Matrix);
}
template<typename Type>
/* Uploads unit to the given location on the CUDA device (returns true, if successful; needs RAW data address) */
inline bool Matrix<Type>::uploadAt(Matrix *address)const {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_AT_BODY(Matrix);
}
template<typename Type>
/* Uploads given source array/unit to the given target location on CUDA device (returns true, if successful; needs RAW data address) */
inline bool Matrix<Type>::upload(const Matrix *source, Matrix *target, int count) {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_AT_BODY(Matrix);
}
template<typename Type>
/* Uploads given source array/unit to CUDA device and returns the clone address */
inline Matrix<Type>* Matrix<Type>::upload(const Matrix<Type> *source, int count) {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_BODY(Matrix);
}
template<typename Type>
/* Disposed given array/unit on CUDA device, making it ready to be free-ed (returns true, if successful) */
inline bool Matrix<Type>::dispose(Matrix *arr, int count) {
	IMPLEMENT_CUDA_LOAD_INTERFACE_DISPOSE_BODY(Matrix);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<typename ElemType>
__device__ __host__ inline void TypeTools<Matrix<ElemType> >::init(Matrix<ElemType> &m){
	TypeTools<Stacktor<ElemType, 1> >::init(m.data);
	m.matWidth = 0;
	m.matHeight = 0;
}
template<typename ElemType>
__device__ __host__ inline void TypeTools<Matrix<ElemType> >::dispose(Matrix<ElemType> &m){
	TypeTools<Stacktor<ElemType, 1> >::dispose(m.data);
}
template<typename ElemType>
__device__ __host__ inline void TypeTools<Matrix<ElemType> >::swap(Matrix<ElemType> &a, Matrix<ElemType> &b){
	TypeTools<Stacktor<ElemType, 1> >::swap(a.data, b.data);
	TypeTools<int>::swap(a.matWidth, b.matWidth);
	TypeTools<int>::swap(a.matHeight, b.matHeight);
}
template<typename ElemType>
__device__ __host__ inline void TypeTools<Matrix<ElemType> >::transfer(Matrix<ElemType> &src, Matrix<ElemType> &dst){
	TypeTools<ElemType>::transfer(src.data, dst.data);
	dst.matWidth = src.matWidth;
	dst.matHeight = src.matHeight;
}

template<typename ElemType>
inline bool TypeTools<Matrix<ElemType> >::prepareForCpyLoad(const Matrix<ElemType> *source, Matrix<ElemType> *hosClone, Matrix<ElemType> *devTarget, int count){
	int i = 0;
	for (i = 0; i < count; i++){
		if (!TypeTools<Stacktor<ElemType, 1> >::prepareForCpyLoad(&(source + i)->data, &(hosClone + i)->data, &(devTarget + i)->data, 1)) break;
		hosClone[i].matWidth = source[i].matWidth;
		hosClone[i].matHeight = source[i].matHeight;
	}
	if (i < count){
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return(false);
	}
	return(true);
}

template<typename ElemType>
inline void TypeTools<Matrix<ElemType> >::undoCpyLoadPreparations(const Matrix<ElemType> *source, Matrix<ElemType> *hosClone, Matrix<ElemType> *devTarget, int count){
	for (int i = 0; i < count; i++)
		TypeTools<Stacktor<ElemType, 1> >::undoCpyLoadPreparations(&(source + i)->data, &(hosClone + i)->data, &(devTarget + i)->data, 1);
}

template<typename ElemType>
inline bool TypeTools<Matrix<ElemType> >::devArrayNeedsToBeDisposed(){
	return TypeTools<ElemType>::devArrayNeedsToBeDisposed();
}
template<typename ElemType>
inline bool TypeTools<Matrix<ElemType> >::disposeDevArray(Matrix<ElemType> *arr, int count){
	for (int i = 0; i < count; i++)
		if (!TypeTools<Stacktor<ElemType, 1> >::disposeDevArray(&(arr + i)->data, 1)) return(false);
	return(true);
}

