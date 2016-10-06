#include"MemManip.cuh"


template<typename Type>
// Creates a clone of the given array
__device__ __host__ inline static Type* MemManip::arrayClone(const Type *arr, const int size){
	if (size <= 0) return(NULL);
	Type *newArr = new Type[size];
	if (newArr == NULL) return(NULL);
	for (int i = 0; i < size; i++)
		newArr[i] = arr[i];
	return(newArr);
}

template<typename Type>
// Creates a clone of the given std::vector
inline static Type* MemManip::vectorClone(const std::vector<Type> &vec){
	if (vec.size() <= 0) return(NULL);
	Type *newArr = new Type[vec.size()];
	if (newArr == NULL) return(NULL);
	for (unsigned int i = 0; i < vec.size(); i++)
		newArr[i] = vec[i];
	return(newArr);
}

template<typename Type>
// Swaps the values of the given items
__device__ __host__ inline static void MemManip::swap(Type &a, Type &b){
	Type tmp = a;
	a = b;
	b = tmp;
}

template<typename Type>
// Swaps the content of the given items bit by bit
__device__ __host__ inline static void MemManip::bitwiseSwap(Type &a, Type &b){
	char tmp[sizeof(Type)];
	memcpy(tmp, &a, sizeof(Type));
	memcpy(&a, &b, sizeof(Type));
	memcpy(&b, tmp, sizeof(Type));
}

template<typename Type>
// Transfers the given data from the source to the destination
// Note: If dataTransferFunction is NULL/not specified, operator= will be used.
__device__ __host__ inline static void MemManip::transferData(Type *destination, Type *source, const int size, void(*dataTransferFunction)(Type &dst, Type &src)){
	if (dataTransferFunction != NULL)
		for (int i = 0; i < size; i++)
			dataTransferFunction(destination[i], source[i]);
	else for (int i = 0; i < size; i++) destination[i] = source[i];
}

template<typename Type>
// Transfers the given data from the source to the destination using multiple threads
// Note: If numThreads is equal to 0, 1 thread will be used.
inline static void MemManip::transferDataMultiThread(Type *destination, Type *source, const int size, int numThreads, void(*dataTransferFunction)(Type &dst, Type &src)){
	if (numThreads < 1) numThreads = 1;
	int elemsLeft = size;
	
	std::thread *t = NULL;
	if (numThreads > 1) t = new std::thread[numThreads];
	
	if (t != NULL){
		int chunkSize = size / numThreads;
		for (int i = 0; i < numThreads; i++){
			t[i] = std::thread(transferData<Type>, destination, source, chunkSize, dataTransferFunction);
			elemsLeft -= chunkSize;
			destination += chunkSize;
			source += chunkSize;
		}
	}
	
	if (elemsLeft > 0) transferData<Type>(destination, source, elemsLeft, dataTransferFunction);
	
	if (t != NULL){
		for (int i = 0; i < numThreads; i++) t[i].join();
		delete[] t;
	}
}

template<typename Type>
// Doubles the capacity of the given array
// Notes:	nonDeletableAddr is an optional parameter, to assert that nothing's being deleted, if it's not permitted to;
//			Multithreaded data transfer available only for the host version.
__device__ __host__ inline static bool MemManip::doubleCapacity(Type *&arr, const int elemsUsed, int &currentCapacity, void(*dataTransferFunction)(Type &dst, Type &src), Type *nonDeletableAddr, int numTransferThreads){
	int newCapacity = currentCapacity * 2;
	Type *newArr = new Type[newCapacity];
	if (newArr == NULL) return(false);
	if (arr != NULL){
#ifdef __CUDA_ARCH__
		transferData<Type>(newArr, arr, elemsUsed, dataTransferFunction);
#else
		transferDataMultiThread<Type>(newArr, arr, elemsUsed, numTransferThreads, dataTransferFunction);
#endif
		if (arr != nonDeletableAddr) delete[] arr;
	}
	currentCapacity = newCapacity;
	arr = newArr;
	return(true);
}
