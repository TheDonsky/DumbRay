#include"Matrix.cuh"




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
/** Friends: **/
template<typename ElemType>
__device__ __host__ inline void StacktorTypeTools<Matrix<ElemType> >::init(Matrix<ElemType> &m){
	StacktorTypeTools<Stacktor<Type, 1> >::init(m.data);
	m.matWidth = 0;
	m.matHeight = 0;
}
template<typename ElemType>
__device__ __host__ inline void StacktorTypeTools<Matrix<ElemType> >::dispose(Matrix<ElemType> &m){
	StacktorTypeTools<Stacktor<Type, 1> >::dispose(m.data);
}
template<typename ElemType>
__device__ __host__ inline void StacktorTypeTools<Matrix<ElemType> >::swap(Matrix<ElemType> &a, Matrix<ElemType> &b){
	StacktorTypeTools<Stacktor<Type, 1> >::swap(a.data, b.data);
	StacktorTypeTools<int>::swap(a.matWidth, b.matWidth);
	StacktorTypeTools<int>::swap(a.matHeight, b.matHeight);
}
template<typename ElemType>
__device__ __host__ inline void StacktorTypeTools<Matrix<ElemType> >::transfer(Matrix<ElemType> &src, Matrix<ElemType> &dst){
	StacktorTypeTools<Stacktor<Type, 1> >::transfer(src.data, dst.data);
	dst.matWidth = src.matWidth;
	dst.matHeight = src.matHeight;
}

template<typename ElemType>
inline bool StacktorTypeTools<Matrix<ElemType> >::prepareForCpyLoad(const Matrix<ElemType> *source, Matrix<ElemType> *hosClone, Matrix<ElemType> *devTarget, int count){
	int i = 0;
	for (i = 0; i < count; i++){
		if (!StacktorTypeTools<Stacktor<Type, 1> >::prepareForCpyLoad(&(source + i)->data, &(hosClone + i)->data, &(devTarget + i)->data, 1)) break;
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
inline void StacktorTypeTools<Matrix<ElemType> >::undoCpyLoadPreparations(const Matrix<ElemType> *source, Matrix<ElemType> *hosClone, Matrix<ElemType> *devTarget, int count){
	for (int i = 0; i < count; i++)
		StacktorTypeTools<Stacktor<Type, 1> >::undoCpyLoadPreparations(&(source + i)->data, &(hosClone + i)->data, &(devTarget + i)->data, 1);
}

template<typename ElemType>
inline bool StacktorTypeTools<Matrix<ElemType> >::devArrayNeedsToBeDisoposed(){
	return StacktorTypeTools<Stacktor<Type, 1> >::devArrayNeedsToBeDisoposed();
}
template<typename ElemType>
inline bool StacktorTypeTools<Matrix<ElemType> >::disposeDevArray(Matrix<ElemType> *arr, int count){
	for (int i = 0; i < count; i++)
		if (!StacktorTypeTools<Stacktor<Type, 1> >::disposeDevArray(&(arr + i)->data, 1)) return(false);
	return(true);
}

