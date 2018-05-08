#include"Generic.cuh"







/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Construction/Destruction/copy-construction: **/
template<typename FunctionPack>
__device__ __host__ inline Generic<FunctionPack>::Generic() {
	memoryManagementModule.isClone = false;
	setNULL();
}
template<typename FunctionPack>
__device__ __host__ inline Generic<FunctionPack>::~Generic() {
	clean();
}
template<typename FunctionPack>
__device__ __host__ inline bool Generic<FunctionPack>::clean() {
	bool cleaned = (memoryManagementModule.disposeFunction == NULL);
	if (!cleaned) cleaned = (memoryManagementModule.disposeFunction(*this));
	if (cleaned) {
		setNULL();
		return true;
	}
	else return false;
}

template<typename FunctionPack>
__host__ inline Generic<FunctionPack>::Generic(const Generic& g) : Generic() {
	copyFrom(g);
}
template<typename FunctionPack>
__host__ inline Generic<FunctionPack>& Generic<FunctionPack>::operator=(const Generic& g) {
	copyFrom(g);
	return (*this);
}
template<typename FunctionPack>
__host__ inline bool Generic<FunctionPack>::copyFrom(const Generic& g) {
	if ((&g) == this) return true;
	if (!clean()) return false;
	if (g.memoryManagementModule.copyFunction != NULL)
		return g.memoryManagementModule.copyFunction(g, (*this));
	else return true;
}

template<typename FunctionPack>
__device__ __host__ inline Generic<FunctionPack>::Generic(Generic&& g) : Generic() {
	swapWith(g);
}
template<typename FunctionPack>
__device__ __host__ inline Generic<FunctionPack>& Generic<FunctionPack>::operator=(const Generic&& g) {
	swapWith(g);
	return (*this);
}
template<typename FunctionPack>
__device__ __host__ inline void Generic<FunctionPack>::swapWith(Generic &g) {
	if ((&g) == this) return;
	TypeTools<volatile void*>::swap(dataPointer, g.dataPointer);
	TypeTools<FunctionPack>::swap(functionPack, g.functionPack);
	TypeTools<MemoryManagementModule>::swap(memoryManagementModule, g.memoryManagementModule);
}

template<typename FunctionPack>
template<typename Type, typename... Args>
__host__ inline Type* Generic<FunctionPack>::use(const Args&... args) {
	clean();
	dataPointer = (volatile void*)(new Type(args...));
	if (dataPointer == NULL) return NULL;
	functionPack.template use<Type>();
	memoryManagementModule.use<Type>();
	return ((Type*)dataPointer);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Getters: **/
template<typename FunctionPack>
__device__ __host__ inline FunctionPack& Generic<FunctionPack>::functions() {
	return functionPack;
}
template<typename FunctionPack>
__device__ __host__ inline const FunctionPack& Generic<FunctionPack>::functions()const {
	return functionPack;
}
template<typename FunctionPack>
__device__ __host__ inline void* Generic<FunctionPack>::object() {
	return (void*)dataPointer;
}
template<typename FunctionPack>
__device__ __host__ inline const void* Generic<FunctionPack>::object()const {
	return (const void*)dataPointer;
}
template<typename FunctionPack>
template<typename Type>
__device__ __host__ inline Type* Generic<FunctionPack>::getObject() {
	return ((Type*)dataPointer);
}
template<typename FunctionPack>
template<typename Type>
__device__ __host__ inline const Type* Generic<FunctionPack>::getObject()const {
	return ((const Type*)dataPointer);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename FunctionPack>
/* Uploads unit to CUDA device and returns the clone address */
inline Generic<FunctionPack>* Generic<FunctionPack>::upload()const {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_BODY(Generic);
}
template<typename FunctionPack>
/* Uploads unit to the given location on the CUDA device (returns true, if successful; needs RAW data address) */
inline bool Generic<FunctionPack>::uploadAt(Generic *address)const {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_AT_BODY(Generic);
}
template<typename FunctionPack>
/* Uploads given source array/unit to the given target location on CUDA device (returns true, if successful; needs RAW data address) */
inline bool Generic<FunctionPack>::upload(const Generic *source, Generic *target, int count) {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_AT_BODY(Generic);
}
template<typename FunctionPack>
/* Uploads given source array/unit to CUDA device and returns the clone address */
inline Generic<FunctionPack>* Generic<FunctionPack>::upload(const Generic<FunctionPack> *source, int count) {
	IMPLEMENT_CUDA_LOAD_INTERFACE_UPLOAD_ARRAY_BODY(Generic);
}
template<typename FunctionPack>
/* Disposed given array/unit on CUDA device, making it ready to be free-ed (returns true, if successful) */
inline bool Generic<FunctionPack>::dispose(Generic *arr, int count) {
	IMPLEMENT_CUDA_LOAD_INTERFACE_DISPOSE_BODY(Generic);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Memory Management Module: **/
template<typename FunctionPack>
__device__ __host__ inline void Generic<FunctionPack>::MemoryManagementModule::useNULLs() {
	copyFunction = NULL;
	disposeFunction = NULL;
	prepareForCpyLoadFunction = NULL;
}
template<typename FunctionPack>
template<typename Type>
__host__ inline void Generic<FunctionPack>::MemoryManagementModule::use() {
	copyFunction = copy<Type>;
	disposeFunction = dispose<Type>;
	prepareForCpyLoadFunction = prepareForCpyLoad<Type>;
}

template<typename FunctionPack>
template<typename Type>
__host__ inline bool Generic<FunctionPack>::MemoryManagementModule::copy(const Generic &source, Generic &destination) {
	if ((&source) == (&destination)) return true;
	return (destination.use<Type>(*((Type*)source.dataPointer)) != NULL);
}
template<typename FunctionPack>
template<typename Type>
__device__ __host__ inline bool Generic<FunctionPack>::MemoryManagementModule::dispose(Generic &g) {
	if (g.dataPointer != NULL) {
		delete g.getObject<Type>();
		g.dataPointer = NULL;
	}
	return true;
}


namespace GenericPrivateKernels {
	template<typename FunctionPack, typename Type>
	__global__ static void configureFunctions(FunctionPack *functions) {
		functions->template use<Type>();
	}
}

template<typename FunctionPack>
template<typename Type>
__host__ inline bool Generic<FunctionPack>::MemoryManagementModule::prepareForCpyLoad(const Generic *source, Generic *hosClone, Generic *devTarget, cudaStream_t &stream) {
	hosClone->memoryManagementModule.isClone = true;
	if (source->dataPointer == NULL) {
		hosClone->setNULL();
		return true;
	}
	
	FunctionPack *functions = (&devTarget->functionPack);
	GenericPrivateKernels::configureFunctions<FunctionPack, Type><<<1, 1, 0, stream>>>(functions);
	if (cudaStreamSynchronize(stream) != cudaSuccess) return false;

	if (cudaMemcpyAsync(&hosClone->functionPack, functions, sizeof(FunctionPack), cudaMemcpyDeviceToHost, stream) != cudaSuccess) return false;
	if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
	
	if (cudaMalloc(&hosClone->dataPointer, sizeof(Type)) != cudaSuccess) return false;
	char hostCloneDummy[sizeof(Type)];
	
	bool success = TypeTools<Type>::prepareForCpyLoad(source->getObject<Type>(), ((Type*)hostCloneDummy), hosClone->getObject<Type>(), 1);
	if (success) {
		success = (cudaMemcpyAsync((void*)hosClone->dataPointer, (void*)hostCloneDummy, sizeof(Type), cudaMemcpyHostToDevice, stream) == cudaSuccess);
		if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;
		if (!success) TypeTools<Type>::undoCpyLoadPreparations(source->getObject<Type>(), ((Type*)hostCloneDummy), hosClone->getObject<Type>(), 1);
	}
	
	if (!success) {
		cudaFree((void*)hosClone->dataPointer);
		hosClone->dataPointer = NULL;
	}
	else {
		hosClone->memoryManagementModule.useNULLs();
		hosClone->memoryManagementModule.disposeFunction = disposeOnDevice<Type>;
	}
	return success;
}

template<typename FunctionPack>
template<typename Type>
__host__ inline bool Generic<FunctionPack>::MemoryManagementModule::disposeOnDevice(Generic &g) {
	if (!g.memoryManagementModule.isClone) return false;
	if (g.dataPointer != NULL) {
		if (!TypeTools<Type>::disposeDevArray(g.getObject<Type>(), 1)) return false;
		if (cudaFree((void*)g.dataPointer) != cudaSuccess) return false;
		g.dataPointer = NULL;
		g.setNULL();
	}
	return true;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Private functions: **/
template<typename FunctionPack>
__device__ __host__ inline void Generic<FunctionPack>::setNULL() {
	memoryManagementModule.useNULLs();
	functionPack.clean();
	dataPointer = NULL;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<typename FunctionPack>
__device__ __host__ inline void TypeTools<Generic<FunctionPack> >::init(Generic<FunctionPack> &m) {
	m.memoryManagementModule.isClone = false;
	m.setNULL();
}
template<typename FunctionPack>
__device__ __host__ inline void TypeTools<Generic<FunctionPack> >::dispose(Generic<FunctionPack> &m) {
	if (!m.memoryManagementModule.isClone) m.clean();
}
template<typename FunctionPack>
__device__  inline void TypeTools<Generic<FunctionPack> >::swap(Generic<FunctionPack> &a, Generic<FunctionPack> &b) {
	a.swapWith(b);
}
template<typename FunctionPack>
__device__ __host__ inline void TypeTools<Generic<FunctionPack> >::transfer(Generic<FunctionPack> &src, Generic<FunctionPack> &dst) {
	src.swapWith(dst);
}

template<typename FunctionPack>
inline bool TypeTools<Generic<FunctionPack> >::prepareForCpyLoad(const Generic<FunctionPack> *source, Generic<FunctionPack> *hosClone, Generic<FunctionPack> *devTarget, int count) {
	cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
	int i = 0;
	for (i = 0; i < count; i++) {
		if (source[i].memoryManagementModule.prepareForCpyLoadFunction != NULL) {
			if (!source[i].memoryManagementModule.prepareForCpyLoadFunction((source + i), (hosClone + i), (devTarget + i), stream)) break;
		}
		else {
			hosClone[i].memoryManagementModule.isClone = true;
			hosClone[i].setNULL();
		}
	}
	bool success = (i >= count);
	success &= (cudaStreamDestroy(stream) == cudaSuccess);
	if (!success) undoCpyLoadPreparations(source, hosClone, devTarget, i);
	return success;
}

template<typename FunctionPack>
inline void TypeTools<Generic<FunctionPack> >::undoCpyLoadPreparations(const Generic<FunctionPack> *, Generic<FunctionPack> *hosClone, Generic<FunctionPack> *, int count) {
	for (int i = 0; i < count; i++)
		hosClone[i].clean();
}

template<typename FunctionPack>
inline bool TypeTools<Generic<FunctionPack> >::devArrayNeedsToBeDisposed() {
	return true;
}
template<typename FunctionPack>
inline bool TypeTools<Generic<FunctionPack> >::disposeDevArray(Generic<FunctionPack> *arr, int count) {
	cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
	char localBuffer[32 * sizeof(Generic<FunctionPack>)];
	char *garbage = NULL;
	Generic<FunctionPack> *hosClone = (Generic<FunctionPack>*)localBuffer;
	if (count > 32) {
		garbage = new char[count * sizeof(Generic<FunctionPack>)];
		if (garbage == NULL) {
			cudaStreamDestroy(stream);
			return false;
		}
	}
	bool success = (cudaMemcpyAsync(hosClone, arr, (sizeof(Generic<FunctionPack>) * count), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
	if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;
	if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	if (success) for (int i = 0; i < count; i++)
		if (!hosClone[i].clean()) success = false;
	if (garbage != NULL) delete[] garbage;
	return success;
}


