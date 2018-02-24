#pragma once

#include"cuda_runtime.h"
#include"device_launch_parameters.h"



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/*
This is the content, required for the Stacktor(and other classes) 
to be able to transfer data between host and device;
Default one will work fine with primitive(fully stack-contained) types an Stacktors,
but will need to be changed/overloaded in case of more complex structures/classes,
if the user is going to upload anything on CUDA device, or initialise from raw data.
*/
#define DEFINE_TYPE_TOOLS_CONTENT_FOR(ElemType) \
	public: \
		typedef ElemType Type; \
		\
		/* ################## init ################# */ \
		/* init should be able to initialise RAW data (it basically has to call a constructor/equivalent) */ \
		/* (default: nothing) */ \
		/* (return value: nothing) */ \
		__device__ __host__ inline static void init(Type &t); \
		\
		/* ################## dispose ################# */ \
		/* dispose should be able to clear the given object, and make it safe to free thre allocation */ \
		/* (default: nothing) */ \
		/* (return value: nothing) */ \
		__device__ __host__ inline static void dispose(Type &t); \
		\
		/* ################## swap ################# */ \
		/* swap should swap given values */ \
		/* (default: { tmp = a; a = b; b = tmp; } ) */ \
		/* (return value: nothing) */ \
		__device__ __host__ inline static void swap(Type &a, Type &b); \
		\
		/* ################## transfer ################# */ \
		/* transfer should transfer the source value to destination */ \
		/* (default: { dst = src; } ) */ \
		/* (return value: nothing) */ \
		__device__ __host__ inline static void transfer(Type &src, Type &dst); \
		\
		\
		/* ################## prepareForCpyLoad ################# */ \
		/* prepareForCpyLoad should prepare hosClone to become a clone of source, after it's uploaded to devTarget using cudaMemcpy() */ \
		/* (default: { for(int i = 0; i < count; i++) hosClone[i] = source[i]; } ) */ \
		/* (return value: true, if successful) */ \
		inline static bool prepareForCpyLoad(const Type *source, Type *hosClone, Type *devTarget, int count); \
		\
		/* ################## undoCpyLoadPreparations ################# */ \
		/* undoCpyLoadPreparations should undo anything that prepareForCpyLoad did (has to clean external allocation and is called in case something failed) */ \
		/* (default: nothing) */ \
		/* (return value: nothing) */ \
		inline static void undoCpyLoadPreparations(const Type *source, Type *hosClone, Type *devTarget, int count); \
		\
		/* ################## devArrayNeedsToBeDisposed ################# */ \
		/* devArrayNeedsToBeDisposed tells, if the device array needs to be disposed before deallocation */ \
		/* (default: false) */ \
		/* (return value: true, if calling didposeDevArray makes sence) */ \
		inline static bool devArrayNeedsToBeDisposed(); \
		\
		/* ################## disposeDevArray ################# */ \
		/* disposeDevArray should dispose the array on the device(without deallocation it) */ \
		/* (default: nothing) */ \
		/* (return value: true, if successful) */ \
		inline static bool disposeDevArray(Type *arr, int count)

#define DEFINE_TYPE_TOOLS_FOR(ElemType) \
	template<> \
	class TypeTools<ElemType>{ \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(ElemType); \
	}

#define DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(ElemType) \
	__device__ __host__ inline friend void TypeTools<ElemType>::init(ElemType &t); \
	__device__ __host__ inline friend void TypeTools<ElemType>::dispose(ElemType &t); \
	__device__ __host__ inline friend void TypeTools<ElemType>::swap(ElemType &a, ElemType &b); \
	__device__ __host__ inline friend void TypeTools<ElemType>::transfer(ElemType &src, ElemType &dst); \
	\
	inline friend bool TypeTools<ElemType>::prepareForCpyLoad(const ElemType *source, ElemType *hosClone, ElemType *devTarget, int count); \
	inline friend void TypeTools<ElemType>::undoCpyLoadPreparations(const ElemType *source, ElemType *hosClone, ElemType *devTarget, int count); \
	inline friend bool TypeTools<ElemType>::devArrayNeedsToBeDisposed(); \
	inline friend bool TypeTools<ElemType>::disposeDevArray(ElemType *arr, int count)

#define SPECIALISE_TYPE_TOOLS_FOR(ElemType) \
	template<> __device__ __host__ inline void TypeTools<ElemType>::init(ElemType &t); \
	template<> __device__ __host__ inline void TypeTools<ElemType>::dispose(ElemType &t); \
	template<> __device__ __host__ inline void TypeTools<ElemType>::swap(ElemType &a, ElemType &b); \
	template<> __device__ __host__ inline void TypeTools<ElemType>::transfer(ElemType &src, ElemType &dst); \
	\
	template<> inline bool TypeTools<ElemType>::prepareForCpyLoad(const ElemType *source, ElemType *hosClone, ElemType *devTarget, int count); \
	template<> inline void TypeTools<ElemType>::undoCpyLoadPreparations(const ElemType *source, ElemType *hosClone, ElemType *devTarget, int count); \
	template<> inline bool TypeTools<ElemType>::devArrayNeedsToBeDisposed(); \
	template<> inline bool TypeTools<ElemType>::disposeDevArray(ElemType *arr, int count)





template<typename Type>
class TypeTools{
	DEFINE_TYPE_TOOLS_CONTENT_FOR(Type);
};





#define DEFINE_CUDA_LOAD_INTERFACE_FOR(Type) \
	/* Uploads unit to CUDA device and returns the clone address */ \
	inline Type* upload()const; \
	/* Uploads unit to the given location on the CUDA device (returns true, if successful; needs RAW data address) */ \
	inline bool uploadAt(Type *address)const; \
	/* Uploads given source array/unit to the given target location on CUDA device (returns true, if successful; needs RAW data address) */ \
	inline static bool upload(const Type *source, Type *target, int count = 1); \
	/* Uploads given source array/unit to CUDA device and returns the clone address */ \
	inline static Type* upload(const Type *source, int count = 1); \
	/* Disposed given array/unit on CUDA device, making it ready to be free-ed (returns true, if successful) */ \
	inline static bool dispose(Type *arr, int count = 1)



#define IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_BODY(Type) return upload(this, 1)
#define IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD(Type) \
	/* Uploads unit to CUDA device and returns the clone address */ \
	inline Type* Type::upload()const{ \
		IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_BODY(Type); \
	}
#define IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_AT_BODY(Type) return upload(this, address, 1)
#define IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_AT(Type) \
	/* Uploads unit to the given location on the CUDA device (returns true, if successful; needs RAW data address) */ \
	inline bool Type::uploadAt(Type *address)const{ \
		IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_AT_BODY(Type); \
	}
#define IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_AT_BODY(Type) \
	if(source == NULL || target == NULL) return false; \
	if(count < 1) count = 1; \
	char stackJunk[sizeof(Type)]; \
	char *heapJunk = NULL; if(count > 1){ heapJunk = new char[sizeof(Type) * count]; if(heapJunk == NULL) return false; } \
	Type *hosClone; if(count > 1) hosClone = ((Type*)heapJunk); else hosClone = ((Type*)stackJunk); \
	bool success = TypeTools<Type>::prepareForCpyLoad(source, hosClone, target, count); \
	if(success){ \
		cudaStream_t stream; success = (cudaStreamCreate(&stream) == cudaSuccess); \
		if(success){ \
			success = (cudaMemcpyAsync(target, hosClone, sizeof(Type) * count, cudaMemcpyHostToDevice, stream) == cudaSuccess); \
			if(success) success = (cudaStreamSynchronize(stream) == cudaSuccess); \
			if(cudaStreamDestroy(stream) != cudaSuccess) success = false; \
						} \
			} \
	if(heapJunk != NULL) delete[] heapJunk; \
	return success
#define IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_AT(Type) \
	/* Uploads given source array/unit to the given target location on CUDA device (returns true, if successful; needs RAW data address) */ \
	inline bool Type::upload(const Type *source, Type *target, int count){ \
		IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_AT_BODY(Type); \
	}
#define IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_BODY(Type) \
	if (count < 1) count = 1; \
	Type *target; if(cudaMalloc(&target, sizeof(Type) * count) != cudaSuccess) return NULL; \
	if (upload(source, target, count)) return target; \
	cudaFree(target); return NULL
#define IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY(Type) \
	/* Uploads given source array/unit to CUDA device and returns the clone address */ \
	inline Type* Type::upload(const Type *source, int count){ \
		IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_BODY(Type); \
	}
#define IMPLEMENT_CUDA_LOAD_INTERFACE_DISPOSE_BODY(Type) \
	if(arr == NULL) return false; \
	if(count < 1) count = 1; \
	return TypeTools<Type>::disposeDevArray(arr, count)
#define IMPLEMENT_CUDA_LOAD_INTERFACE_DISPOSE(Type) \
	/* Disposed given array/unit on CUDA device, making it ready to be free-ed (returns true, if successful) */ \
	inline bool Type::dispose(Type *arr, int count){ \
		IMPLEMENT_CUDA_LOAD_INTERFACE_DISPOSE_BODY(Type); \
	}

#define IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(Type) \
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD(Type) \
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_AT(Type) \
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_AT(Type) \
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY(Type) \
	IMPLEMENT_CUDA_LOAD_INTERFACE_DISPOSE(Type)


#define IMPLEMENT_CUDA_LOAD_INTERFACE_FOR_TEMPLATE(Type) \
	template<typename TemplateType> \
	/* Uploads unit to CUDA device and returns the clone address */ \
	inline Type<TemplateType>* Type<TemplateType>::upload()const { \
		IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_BODY(Type); \
	} \
	template<typename TemplateType> \
	/* Uploads unit to the given location on the CUDA device (returns true, if successful; needs RAW data address) */ \
	inline bool Type<TemplateType>::uploadAt(Type *address)const { \
		IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_AT_BODY(Type); \
	} \
	template<typename TemplateType> \
	/* Uploads given source array/unit to the given target location on CUDA device (returns true, if successful; needs RAW data address) */ \
	inline bool Type<TemplateType>::upload(const Type *source, Type *target, int count) { \
		IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_AT_BODY(Type); \
	} \
	template<typename TemplateType> \
	/* Uploads given source array/unit to CUDA device and returns the clone address */ \
	inline Type<TemplateType>* Type<TemplateType>::upload(const Type<TemplateType> *source, int count) { \
		IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_BODY(Type); \
	} \
	template<typename TemplateType> \
	/* Disposed given array/unit on CUDA device, making it ready to be free-ed (returns true, if successful) */ \
	inline bool Type<TemplateType>::dispose(Type *arr, int count) { \
		IMPLEMENT_CUDA_LOAD_INTERFACE_DISPOSE_BODY(Type); \
	}



#define COPY_TYPE_TOOLS_IMPLEMENTATION(Child, Perent) \
	template<> __device__ __host__ inline void TypeTools<Child>::init(Child &m) { \
		TypeTools<Perent >::init(m); \
	} \
	template<> __device__ __host__ inline void TypeTools<Child>::dispose(Child &m) { \
		TypeTools<Perent >::dispose(m); \
	} \
	template<> __device__ __host__ inline void TypeTools<Child>::swap(Child &a, Child &b) { \
		TypeTools<Perent >::swap(a, b); \
	} \
	template<> __device__ __host__ inline void TypeTools<Child>::transfer(Child &src, Child &dst) { \
		TypeTools<Perent >::transfer(src, dst); \
	} \
	template<> inline bool TypeTools<Child>::prepareForCpyLoad(const Child *source, Child *hosClone, Child *devTarget, int count) { \
		return TypeTools<Perent >::prepareForCpyLoad(source, hosClone, devTarget, count); \
	} \
	template<> inline void TypeTools<Child>::undoCpyLoadPreparations(const Child *source, Child *hosClone, Child *devTarget, int count) { \
		TypeTools<Perent >::undoCpyLoadPreparations(source, hosClone, devTarget, count); \
	} \
	template<> inline bool TypeTools<Child>::devArrayNeedsToBeDisposed() { \
		return TypeTools<Perent >::devArrayNeedsToBeDisposed(); \
	} \
	template<> inline bool TypeTools<Child>::disposeDevArray(Child *arr, int count) { \
		return TypeTools<Perent >::disposeDevArray(arr, count); \
	}












/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/* Default implementations: */
template<typename Type>
__device__ __host__ inline void TypeTools<Type>::init(Type &) {}
template<typename Type>
__device__ __host__ inline void TypeTools<Type>::dispose(Type &) {}
template<typename Type>
__device__ __host__ inline void TypeTools<Type>::swap(Type &a, Type &b) {
	Type c = a;
	a = b;
	b = c;
}
template<typename Type>
__device__ __host__ inline void TypeTools<Type>::transfer(Type &src, Type &dst) {
	dst = src;
}

template<typename Type>
inline bool TypeTools<Type>::prepareForCpyLoad(const Type *source, Type *hosClone, Type *, int count) {
	for (int i = 0; i < count; i++)
		hosClone[i] = source[i];
	return(true);
}
template<typename Type>
inline void TypeTools<Type>::undoCpyLoadPreparations(const Type *, Type *, Type *, int) { }
template<typename Type>
inline bool TypeTools<Type>::devArrayNeedsToBeDisposed() { return(false); }
template<typename Type>
inline bool TypeTools<Type>::disposeDevArray(Type *, int) { return(true); }





#include"TypeTools.impl.cuh"
