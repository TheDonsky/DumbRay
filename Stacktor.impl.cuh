#include"Stacktor.cuh"


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename ElemType, unsigned int localCapacity>
__device__ __host__ inline void TypeTools<Stacktor<ElemType, localCapacity> >::init(Type &t){
	t.initRaw();
}
template<typename ElemType, unsigned int localCapacity>
__device__ __host__ inline void TypeTools<Stacktor<ElemType, localCapacity> >::dispose(Type &t){
	t.disposeRaw();
}
template<typename ElemType, unsigned int localCapacity>
__device__ __host__ inline void TypeTools<Stacktor<ElemType, localCapacity> >::swap(Type &a, Type &b){
	a.swapWith(b);
}
template<typename ElemType, unsigned int localCapacity>
__device__ __host__ inline void TypeTools<Stacktor<ElemType, localCapacity> >::transfer(Type &src, Type &dst){
#ifdef __CUDA_ARCH__
	if (((Type*)(src.clone.address)) != NULL && ((Type*)(dst.clone.address)) != NULL)
		dst = src;
#endif
	src.swapWith(dst);
}

template<typename ElemType, unsigned int localCapacity>
inline bool TypeTools<Stacktor<ElemType, localCapacity> >::prepareForCpyLoad(const Type *source, Type *hosClone, Type *devTarget, int count){
	return(Type::upload(source, hosClone, devTarget, count, true));
}
template<typename ElemType, unsigned int localCapacity>
inline void TypeTools<Stacktor<ElemType, localCapacity> >::undoCpyLoadPreparations(const Type *source, Type *hosClone, Type *devTarget, int count){
	for (int i = 0; i < count; i++)
		TypeTools<ElemType>::undoCpyLoadPreparations(source[i].stackData, hosClone[i].stackData, (devTarget + i)->stackData, localCapacity);
	if (hosClone[0].clone.address != NULL){
		Type::disposeOfExternalData(devTarget, hosClone, count, 0);
		cudaFree(hosClone[0].clone.address);
	}
}
template<typename ElemType, unsigned int localCapacity>
inline bool TypeTools<Stacktor<ElemType, localCapacity> >::devArrayNeedsToBeDisposed(){ return(true); }
template<typename ElemType, unsigned int localCapacity>
inline bool TypeTools<Stacktor<ElemType, localCapacity> >::disposeDevArray(Type *arr, int count){
	return(Type::dispose(arr, count));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** ########################################################################## **/
/** Stacktor:                                                                  **/
/** ########################################################################## **/





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Construction/Destruction/copy-construction: **/


/** ========================================================== **/
/*| Regular |*/

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
// Simple constructor (constructs empty)
__device__ __host__ inline Stacktor<Type, localCapacity>::Stacktor(){
	init(false);
}
template<typename Type, unsigned int localCapacity>
// Destructor
__device__ __host__ inline Stacktor<Type, localCapacity>::~Stacktor(){
	dispose(false);
}

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
template<typename... Elems>
// Constructs and fills with given elements
__device__ __host__ inline Stacktor<Type, localCapacity>::Stacktor(const Type &elem, const Elems&... elems){
	init(false);
	push(elem, elems...);
}

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
template<unsigned int arrSize>
// Constructs and copies the given fixed size array to the Stacktor
__device__ __host__ inline Stacktor<Type, localCapacity>::Stacktor(const Type(&arr)[arrSize]){
	init(false);
	(*this) = arr;
}
template<typename Type, unsigned int localCapacity>
template<unsigned int arrSize>
// Makes the Stacktor equivelent of the given fixed size array
__device__ __host__ inline Stacktor<Type, localCapacity>& Stacktor<Type, localCapacity>::operator=(const Type(&arr)[arrSize]){
	clear();
	return((*this) += arr);
}

/** ========================================================== **/
/*| Raw data |*/
template<typename Type, unsigned int localCapacity>
// Constructor for raw data (note: never use this on anything allocated on stack, or using new)
__device__ __host__ inline void Stacktor<Type, localCapacity>::initRaw(){
	init(true);
}
template<typename Type, unsigned int localCapacity>
// Destructor for raw data (note: never use this on anything allocated on stack, or using new)
__device__ __host__ inline void Stacktor<Type, localCapacity>::disposeRaw(){
	dispose(true);
}


/** ========================================================== **/
/*| Copy-construction |*/
template<typename Type, unsigned int localCapacity>
// Copy-constructor
__device__ __host__ inline Stacktor<Type, localCapacity>::Stacktor(const Stacktor &s){
	init(false);
	(*this) = s;
}
template<typename Type, unsigned int localCapacity>
// Operator = (copies the data)
__device__ __host__ inline Stacktor<Type, localCapacity>& Stacktor<Type, localCapacity>::operator=(const Stacktor &s){
	if (this == (&s)) return (*this);
	clear();
	demandCapacity(s.allocated);
	for (int i = 0; i < s.used; i++)
		data[i] = s.data[i];
	used = s.used;
	return(*this);
}
// Pass-constructor
template<typename Type, unsigned int localCapacity>
__device__ __host__ inline Stacktor<Type, localCapacity>::Stacktor(Stacktor &&s) : Stacktor(){
	this->swapWith(s);
}
template<typename Type, unsigned int localCapacity>
// Pass-assignment
__device__ __host__ inline Stacktor<Type, localCapacity>& Stacktor<Type, localCapacity>::operator=(Stacktor &&s) {
	this->swapWith(s);
	return (*this);
}
template<typename Type, unsigned int localCapacity>
// Swaps the content with the given Stacktor
__device__ __host__ inline void Stacktor<Type, localCapacity>::swapWith(Stacktor &s){
#ifdef __CUDA_ARCH__
	if (((Type*)(clone.address)) != NULL && ((Type*)(s.clone.address)) != NULL){
		Stacktor tmp = (*this);
		(*this) = s;
		s = tmp;
		return;
	}
#endif
	if (data == stackData || s.data == s.stackData)
		for (int i = 0; i < localCapacity; i++)
			TypeTools<Type>::swap(stackData[i], s.stackData[i]);

	Type *tmpData = data;
	if (s.data != s.stackData) data = s.data;
	else data = stackData;
	if (tmpData != stackData) s.data = tmpData;
	else s.data = s.stackData;
#ifdef __CUDA_ARCH__
	if ((Type*)(s.clone.address) == data) clone.address = data;
	if ((Type*)(clone.address) == s.data) s.clone.address = s.data;
#endif

	TypeTools<int>::swap(used, s.used);
	TypeTools<int>::swap(allocated, s.allocated);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Data manipulation: **/


/** ========================================================== **/
/*| Insert/remove/clear |*/

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
// Adds the given element to the Stacktor
__device__ __host__ inline void Stacktor<Type, localCapacity>::push(const Type &elem){
	demandCapacity(used + 1);
	data[used] = elem;
	used++;
}
template<typename Type, unsigned int localCapacity>
template<typename... Rest>
// Adds the given elements to the Stacktor (might be slightly slower)
__device__ __host__ inline void Stacktor<Type, localCapacity>::push(const Type &elem, const Rest&... rest){
	push(elem);
	push(rest...);
}
template<typename Type, unsigned int localCapacity>
// Adds given number of elements to the stacktor
__device__ __host__ inline void Stacktor<Type, localCapacity>::flush(int count){
	demandCapacity(used + count);
	used += count;
}
template<typename Type, unsigned int localCapacity>
template<unsigned int arrSize>
// Adds the elements from the given fixed size array to the Stacktor
__device__ __host__ inline Stacktor<Type, localCapacity>& Stacktor<Type, localCapacity>::operator+=(const Type(&arr)[arrSize]){
	demandCapacity(allocated + arrSize);
	for (int i = 0; i < arrSize; i++)
		push(arr[i]);
	return(*this);
}

template<>
template<unsigned int arrSize>
// Adds the elements from the given fixed size array to the String
__device__ __host__ inline String& String::operator+=(const char(&arr)[arrSize]){
	if (used > 0 && arr[used] == '\0') used--;
	int neededCapacity = used + arrSize;
	demandCapacity(neededCapacity);
	for (unsigned int i = 0; i < arrSize; i++)
		push(arr[i]);
	return(*this);
}

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
// Removes the top element (the last that was added) and returns it
__device__ __host__ inline const Type& Stacktor<Type, localCapacity>::pop(){
	used--;
	return(data[used]);
}
template<typename Type, unsigned int localCapacity>
// Swaps the element with the given index with the last one, then pops and returns it.
__device__ __host__ inline const Type& Stacktor<Type, localCapacity>::swapPop(int index){
	used--;
	TypeTools<Type>::swap(data[index], data[used]);
	return(data[used]);
}
template<typename Type, unsigned int localCapacity>
// Removes the element with the given index and returns it's value
__device__ __host__ inline const Type& Stacktor<Type, localCapacity>::remove(int index){
	Type tmp;
	TypeTools<Type>::transfer(data[index], tmp);
	for (int i = index + 1; i < used; i++)
		TypeTools<Type>::transfer(data[i], data[i - 1]);
	used--;
	TypeTools<Type>::transfer(tmp, data[used]);
	return(data[used]);
}
template<typename Type, unsigned int localCapacity>
// Clears the Stacktor
__device__ __host__ inline void Stacktor<Type, localCapacity>::clear(){
	changeData(stackData, localCapacity);
	used = 0;
}


/** ========================================================== **/
/*| Get/Set |*/

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
// i'th element
__device__ __host__ inline Type& Stacktor<Type, localCapacity>::operator[](int i){
	return(data[i]);
}
template<typename Type, unsigned int localCapacity>
// i'th element (constant)
__device__ __host__ inline const Type& Stacktor<Type, localCapacity>::operator[](int i)const{
	return(data[i]);
}
template<typename Type, unsigned int localCapacity>
// Pointer to the i'th element
__device__ __host__ inline Type* Stacktor<Type, localCapacity>::operator+(int i){
	return(data + i);
}
template<typename Type, unsigned int localCapacity>
// Pointer to the i'th element (constant)
__device__ __host__ inline const Type* Stacktor<Type, localCapacity>::operator+(int i)const{
	return(data + i);
}

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
// The last element
__device__ __host__ inline Type& Stacktor<Type, localCapacity>::top(){
	return(data[used - 1]);
}
template<typename Type, unsigned int localCapacity>
// The last element (constant)
__device__ __host__ inline const Type& Stacktor<Type, localCapacity>::peek()const{
	return(data[used - 1]);
}

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
// Length of the Stacktor
__device__ __host__ inline int Stacktor<Type, localCapacity>::size()const{
	return(used);
}
template<typename Type, unsigned int localCapacity>
// Tells, if Stacktor is empty
__device__ __host__ inline bool Stacktor<Type, localCapacity>::empty()const{
	return(used <= 0);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Upload/download: **/

namespace StacktorPrivateKernels{
#define STACKTOR_KERNELS_THREADS_PER_BLOCK 64
#define STACKTOR_KERNELS_UNITS_PER_THREAD 4

	__device__ inline static int getStartIndex(){
		return(((STACKTOR_KERNELS_THREADS_PER_BLOCK * blockIdx.x) + threadIdx.x) * STACKTOR_KERNELS_UNITS_PER_THREAD);
	}
	__device__ inline static int getEndIndex(int count){
		register int val = (getStartIndex() + STACKTOR_KERNELS_UNITS_PER_THREAD);
		if (val > count) val = count;
		return(val);
	}

	template<typename Type, unsigned int localCapacity>
	__global__ static void clear(Stacktor<Type, localCapacity> *arr, int count){
		int index = getStartIndex();
		int end = getEndIndex(count);
		while (index < end){
			arr[index].clear();
			index++;
		}
	}

	static int getBlockCount(int size){
		register int unitsPerBlock = (STACKTOR_KERNELS_THREADS_PER_BLOCK * STACKTOR_KERNELS_UNITS_PER_THREAD);
		return((size + (unitsPerBlock - 1)) / (unitsPerBlock));
	}
	static int getThreadCount(){
		return(STACKTOR_KERNELS_THREADS_PER_BLOCK);
	}

#undef STACKTOR_KERNELS_THREADS_PER_BLOCK
#undef STACKTOR_KERNELS_UNITS_PER_THREAD
}


/** ========================================================== **/
/*| Units |*/

template<typename Type, unsigned int localCapacity>
/*
Uploads the Stacktor to GPU
Return value:
.	address of the clone (NULL, in case of a failure)
Notes:
.	This will create no "bondage" between the source and the clone;
*/
inline Stacktor<Type, localCapacity>* Stacktor<Type, localCapacity>::upload()const{
	return upload(this, 1);
}

template<typename Type, unsigned int localCapacity>
/*
Uploads the Stacktor to the given address on GPU
Parameters:
.	destination: Location, the Stacktor(s) should be uploaded at;
Return value:
.	true, if successful.
Notes:
.	This will create no "bondage" between the source and the clone;
.	NULL parameter will result in an instant failure.
*/
inline bool Stacktor<Type, localCapacity>::uploadAt(Stacktor *destination)const{
	return upload(this, destination, 1);
}


/** ========================================================== **/
/*| Arrays |*/

/** ------------------------------------ **/

template<typename Type, unsigned int localCapacity>
/*
Uploads the given Stacktor(s) to the given location on GPU
Parameters:
.	source:	the source Stacktor(s);
.	target:	Location, the Stacktor(s) should be uploaded at;
.	count: size of the array (default is 1, so when uploading just one doesn't need to be specified)
Return value:
.	A single boolean, telling if the upload was a success or not (true for successful)
Notes:
.	This will create no "bondage" between the source and the clone;
.	If count is non-positive, it'll be understood as 1;
.	NULL parameters will result in an instant failure.
*/
inline bool Stacktor<Type, localCapacity>::upload(const Stacktor *source, Stacktor *target, int count){
	if (source == NULL || target == NULL) return(false); // Dealing with NULL-s.
	if (count < 1) count = 1; // Making sure, count is not negative or zero.

	/* /////////////////////////////////////////////// */
	/* Allocating CPU clone: */
	char stackGarbage[sizeof(Stacktor<Type, localCapacity>)]; // Locally allocatable size.
	char *garbage = NULL; // Bigger chunk
	Stacktor *hostCopy; // Pointer to whatever gets to play the role of the host clone (will be RAW data).
	if (count > 1){ // Size is greater than what the program can allocate on stack.
		garbage = new char[sizeof(Stacktor<Type, localCapacity>) * count];
		if (garbage == NULL) return(false);
		hostCopy = (Stacktor*)garbage;
	}
	else hostCopy = ((Stacktor*)stackGarbage); // Can be on stack, is the size is small enough.
	
	/* /////////////////////////////////////////////// */
	/* Upload: */
	bool success = upload(source, hostCopy, target, count, false);


	/* /////////////////////////////////////////////// */
	/* Cleaning up and returning: */
	if (garbage != NULL) // External data needs to be disposed.
		delete[] garbage;

	return(success);
}

template<typename Type, unsigned int localCapacity>
/*
Uploads the given Stacktor(s) to GPU and returns the clone's address
Parameters:
.	source:	the source Stacktor(s);
.	count: size of the array (default is 1, so when uploading just one doesn't need to be specified)
Return value:
.	address of the clone (NULL, in case of a failure)
Notes:
.	This will create no "bondage" between the source and the clone;
.	If count is non-positive, it'll be understood as 1;
.	NULL parameters will result in an instant failure.
*/
inline Stacktor<Type, localCapacity>* Stacktor<Type, localCapacity>::upload(const Stacktor *source, int count){
	if (count < 1) count = 1;
	Stacktor *clone; if (cudaMalloc(&clone, sizeof(Stacktor<Type, localCapacity>) * count) != cudaSuccess) return false;
	if (upload(source, clone, count)) return clone;
	else{
		cudaFree(clone);
		return NULL;
	}
}

template<typename Type, unsigned int localCapacity>
/*
Disposes the given DEVICE array, that was initially uploaded
with "upload" function or any similar one from the static part of Stacktor class.
Parameters:
.	arr: the array on the device;
.	count: number of elements (default is 1)
Return value:
.	A single boolean, telling if the disposal was a success or not (true for successful)
Notes:
.	The array will be cleaned and ready to be free-ed, but will stay allocated;
.	The array should be exactly the same one that was initially uploaded,
as for the speed reasons some of the underlying data is interconnected
and can't be deallocated properly;
.	If count is less than 1, it'll be understood as 1.
*/
inline bool Stacktor<Type, localCapacity>::dispose(Stacktor *arr, int count){
	if (count < 1) count = 1;
	cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return(false);

	// Cleanup:
	register int nBlocks = StacktorPrivateKernels::getBlockCount(count);
	register int nThreads = StacktorPrivateKernels::getThreadCount();
	StacktorPrivateKernels::clear<Type, localCapacity><<<nBlocks, nThreads, 0, stream>>>(arr, count);
	
	// Dealling with arrays:
	char junk[sizeof(Stacktor<Type, localCapacity>)];
	Stacktor<Type, localCapacity> *first = (Stacktor<Type, localCapacity>*)junk;
	bool success = (cudaMemcpyAsync(first, arr, sizeof(Stacktor<Type, localCapacity>), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
	if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;

	if (success) success = disposeOfUnderlyingData(arr, NULL, count, (first->clone.address != NULL), stream);

	// Freeing disposed parts:
	if (success && first->clone.address != NULL)
		if (cudaFree(first->clone.address) != cudaSuccess) success = false;
	if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	return(success);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Private helpers: **/


/** ========================================================== **/
/*| Construct/Destruct |*/
template<typename Type, unsigned int localCapacity>
__device__ __host__ inline void Stacktor<Type, localCapacity>::init(bool raw){
	data = stackData;
	used = 0;
	allocated = localCapacity;
	clone.address = NULL;
	clone.size = 0;
	clone.capacity = 0;
	if (raw) for (int i = 0; i < localCapacity; i++)
		TypeTools<Type>::init(stackData[i]);
}
template<typename Type, unsigned int localCapacity>
__device__ __host__ inline void Stacktor<Type, localCapacity>::dispose(bool raw){
	clear();
	if (raw) for (int i = 0; i < localCapacity; i++)
		TypeTools<Type>::dispose(stackData[i]);
}


/** ========================================================== **/
/*| Capacity |*/
template<typename Type, unsigned int localCapacity>
__device__ __host__ inline void Stacktor<Type, localCapacity>::demandCapacity(int newCapacity){
	if (allocated < newCapacity){
		if (newCapacity < allocated * 2)
			newCapacity = (allocated * 2);
		Type *newData = stackData;
		if (newCapacity > localCapacity) newData = new Type[newCapacity];
		if (newData == NULL) return;
		for (int i = 0; i < used; i++)
			TypeTools<Type>::transfer(data[i], newData[i]);
		changeData(newData, newCapacity);
	}
}
template<typename Type, unsigned int localCapacity>
__device__ __host__ inline void Stacktor<Type, localCapacity>::changeData(Type *newData, int newCapacity){
	if (data != stackData && data != ((Type*)clone.address))
		delete[] data;
	data = newData;
	allocated = newCapacity;
}


/** ========================================================== **/
/*| Upload |*/

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
inline bool Stacktor<Type, localCapacity>::upload(const Stacktor *source, Stacktor *hosClone, Stacktor *devTarget, int count, bool dataOnly){
	/* /////////////////////////////////////////////// */
	/* Getting ready: */
	initHosClones(hosClone, devTarget, count); // After this, Stacktor part of the the hosClone array will be fully initiaited as empty (the underlying array will be junk).

	char *exJunk = NULL;
	Type *exAlloc = NULL;
	int exCap = getExternalCapacity(source, count);
	if (exCap > 0){
		exJunk = new char[sizeof(Type) * exCap]; if (exJunk == NULL) return(false);
		if (cudaMalloc((void**)&exAlloc, sizeof(Type) * exCap) != cudaSuccess){ delete[] exJunk; return(false); }
	}
	Type *exHosAlloc = (Type*)exJunk;

	/* /////////////////////////////////////////////// */
	/* Preparing for load: */
	bool success = prepDataForCpyLoad(source, hosClone, devTarget, count, exHosAlloc, exAlloc);

	/* /////////////////////////////////////////////// */
	/* Loading: */
	if (success){
		hosClone[0].clone.address = exAlloc;
		hosClone[0].clone.size = exCap;
		success = cpyLoad(hosClone, devTarget, count, exHosAlloc, exAlloc, exCap, !dataOnly);
		if (!success){
			undoDataLoadPrep(source, hosClone, devTarget, count, exHosAlloc, exAlloc);
			cudaFree(exAlloc);
		}
	}

	/* /////////////////////////////////////////////// */
	/* Cleanup and return: */
	if (exJunk != NULL)
		delete[] exJunk;
	return(success); 
}
template<typename Type, unsigned int localCapacity>
inline void Stacktor<Type, localCapacity>::initHosClones(Stacktor *hosClone, Stacktor *devTarget, int count){
	for (int i = 0; i < count; i++){
		hosClone[i].init(false);
		hosClone[i].data = (devTarget + i)->stackData;
	}
}
template<typename Type, unsigned int localCapacity>
inline int Stacktor<Type, localCapacity>::getExternalCapacity(const Stacktor *source, int count){
	int exCap = 0;
	for (int i = 0; i < count; i++)
		if (source[i].data != source[i].stackData)
			exCap += source[i].allocated;
	return(exCap);
}
template<typename Type, unsigned int localCapacity>
inline bool Stacktor<Type, localCapacity>::prepDataForCpyLoad(const Stacktor *source, Stacktor *hosClone, Stacktor *devTarget, int count, Type *exHosAlloc, Type *exAlloc){
	bool success = true;
	int exUsed = 0;
	int uploaded = 0;
	for (int i = 0; i < count; i++){
		/* Basic preparation: */
		success = TypeTools<Type>::prepareForCpyLoad(source[i].stackData, hosClone[i].stackData, (devTarget + i)->stackData, localCapacity);
		if (!success) break;

		// If external allocation is needed:
		if (source[i].data != source[i].stackData){
			if (!TypeTools<Type>::prepareForCpyLoad(source[i].data, exHosAlloc + exUsed, exAlloc + exUsed, source[i].allocated)){
				// Error occured:
				TypeTools<Type>::undoCpyLoadPreparations(source[i].stackData, hosClone[i].stackData, (devTarget + i)->stackData, localCapacity);
				success = false;
				break;
			}
			else{
				// "Registering" success:
				hosClone[i].data = (exAlloc + exUsed);
				hosClone[i].allocated = source[i].allocated;
				hosClone[i].clone.address = (void*)(hosClone[i].data);
				hosClone[i].clone.capacity = hosClone[i].allocated;
				exUsed += source[i].allocated;
			}
		}
		hosClone[i].used = source[i].used;
		uploaded++;
	}
	/* Failure handling: */
	if (!success) undoDataLoadPrep(source, hosClone, devTarget, uploaded, exHosAlloc, exAlloc);

	return(success);
}
template<typename Type, unsigned int localCapacity>
inline void Stacktor<Type, localCapacity>::undoDataLoadPrep(const Stacktor *source, Stacktor *hosClone, Stacktor *devTarget, int count, Type *exHosAlloc, Type *exAlloc){
	int exUsed = 0;
	for (int i = 0; i < count; i++){
		TypeTools<Type>::undoCpyLoadPreparations(source[i].stackData, hosClone[i].stackData, (devTarget + i)->stackData, localCapacity);
		if (source[i].data != source[i].stackData){
			TypeTools<Type>::undoCpyLoadPreparations(source[i].data, exHosAlloc + exUsed, exAlloc + exUsed, source[i].allocated);
			exUsed += source[i].allocated;
		}
	}
}
template<typename Type, unsigned int localCapacity>
inline bool Stacktor<Type, localCapacity>::cpyLoad(Stacktor *hosClone, Stacktor *devTarget, int count, Type *exHosAlloc, Type *exAlloc, int exCount, bool loadAll){
	cudaStream_t stream[2];
	bool success = true;
	if(exCount > 0) success = (cudaStreamCreate(stream) == cudaSuccess);
	if (success){
		if (loadAll) success = (cudaStreamCreate(stream + 1) == cudaSuccess);
		if (success){
			// Upload:
			if (exCount > 0) success = (cudaMemcpyAsync(exAlloc, exHosAlloc, sizeof(Type) * exCount, cudaMemcpyHostToDevice, stream[0]) == cudaSuccess);
			if (success && loadAll) success = (cudaMemcpyAsync(devTarget, hosClone, sizeof(Stacktor<Type, localCapacity>) * count, cudaMemcpyHostToDevice, stream[1]) == cudaSuccess);
			// Syncronisation:
			if (exCount > 0) if (cudaStreamSynchronize(stream[0]) != cudaSuccess) success = false;
			if (loadAll){
				if (cudaStreamSynchronize(stream[1]) != cudaSuccess) success = false;
				if (cudaStreamDestroy(stream[1]) != cudaSuccess) success = false;
			}
		}
		if (exCount > 0) if (cudaStreamDestroy(stream[0]) != cudaSuccess) success = false;
	}
	return(success);
}

/** ------------------------------------ **/
template<typename Type, unsigned int localCapacity>
inline bool Stacktor<Type, localCapacity>::disposeOfUnderlyingData(Stacktor *arr, Stacktor *devClone, int count, bool hasExternalAllocation, cudaStream_t stream){
	if (!TypeTools<Type>::devArrayNeedsToBeDisposed()) return(true);
	for (int i = 0; i < count; i++)
		if (!TypeTools<Type>::disposeDevArray((arr + i)->stackData, localCapacity)) return(false);
	if (hasExternalAllocation)
		return(disposeOfExternalData(arr, devClone, count, stream));
	else return(true);
}
template<typename Type, unsigned int localCapacity>
inline bool Stacktor<Type, localCapacity>::disposeOfExternalData(Stacktor *arr, Stacktor *hosClone, int count, cudaStream_t stream){
	bool success = true;
	char *junk = NULL;
	if (hosClone == NULL){
		junk = new char[sizeof(Stacktor<Type, localCapacity>) * count];
		if (junk == NULL) return(false);
		hosClone = (Stacktor*)junk;
		success = (cudaMemcpyAsync(hosClone, arr, sizeof(Stacktor<Type, localCapacity>) * count, cudaMemcpyDeviceToHost, stream) == cudaSuccess);
		if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;
	}
	for (int i = 0; i < count && success; i++)
		if (hosClone[i].clone.capacity != 0)
			if (!TypeTools<Type>::disposeDevArray((Type*)(hosClone[i].clone.address), hosClone[i].clone.capacity)){
				success = false;
				break;
			}
	if (junk != NULL) delete[] junk;
	return(success);
}

