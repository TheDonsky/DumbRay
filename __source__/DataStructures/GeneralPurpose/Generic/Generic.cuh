#pragma once
#include"../Stacktor/Stacktor.cuh"




/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename FunctionPack> class Generic;
template<typename FunctionPack>
class TypeTools<Generic<FunctionPack> > {
public:
	typedef Generic<FunctionPack> GenericType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(GenericType);
};




template<typename FunctionPack>
class Generic {
public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Construction/Destruction/copy-construction: **/
	__device__ __host__ inline Generic();
	__device__ __host__ inline ~Generic();
	__device__ __host__ inline bool clean();
	
	__host__ inline Generic(const Generic& g);
	__host__ inline Generic& operator=(const Generic& g);
	__host__ inline bool copyFrom(const Generic& g);

	__device__ __host__ inline Generic(Generic&& g);
	__device__ __host__ inline Generic& operator=(const Generic&& g);
	__device__ __host__ inline void swapWith(Generic &g);
	
	template<typename Type, typename... Args>
	__host__ inline Type* use(const Args&... args);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Getters: **/
	__device__ __host__ inline FunctionPack& functions();
	__device__ __host__ inline const FunctionPack& functions()const;
	__device__ __host__ inline void* object();
	__device__ __host__ inline const void* object()const;
	template<typename Type>
	__device__ __host__ inline Type* getObject(); 
	template<typename Type>
	__device__ __host__ inline const Type* getObject()const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	DEFINE_CUDA_LOAD_INTERFACE_FOR(Generic);





private:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Parameters: **/
	volatile void *dataPointer;
	FunctionPack functionPack;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Memory Management Module: **/
	struct MemoryManagementModule {
		bool(*copyFunction)(const Generic &source, Generic &destination);
		bool(*disposeFunction)(Generic &g);

		bool(*prepareForCpyLoadFunction)(const Generic *source, Generic *hosClone, Generic *devTarget, cudaStream_t &stream);
		bool isClone;

		__device__ __host__ inline void useNULLs();
		template<typename Type>
		__host__ inline void use();

		template<typename Type>
		__host__ inline static bool copy(const Generic &source, Generic &destination);
		template<typename Type>
		__device__ __host__ inline static bool dispose(Generic &g);
		
		template<typename Type>
		__host__ inline static bool prepareForCpyLoad(const Generic *source, Generic *hosClone, Generic *devTarget, cudaStream_t &stream);
		template<typename Type>
		__host__ inline static bool disposeOnDevice(Generic &g);
	} memoryManagementModule;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Private functions: **/
	__device__ __host__ inline void setNULL();





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Friends: **/
	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(Generic);
};







#include"Generic.impl.cuh"
