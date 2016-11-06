#pragma once

#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include"TypeTools.cuh"
#include<iostream>
#include<string>
#include<thread>
#include<math.h>





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename Type, unsigned int localCapacity> class Stacktor;
template<typename ElemType, unsigned int localCapacity>
class TypeTools<Stacktor<ElemType, localCapacity> >{
public:
	typedef Stacktor<ElemType, localCapacity> StacktorType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(StacktorType);
};





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
template<typename Type, unsigned int localCapacity = 1>
/*
	Stacktor class represents a variable sized array, that supports being
	aploaded/downloaded to/from a cuda device.
	Template:
		Type: type of the array.
		localCapacity: the highest size Stacktor can have, without allocating
			anything in heap.
			(Will give a significant speedup, when using arrays of Stacktors,
			if localCapacity is set to their average size during runtime)
	Note:
		TypeTools is a helper class/struct/namespace, that provides Stacktor
		with functionality, that allows it to upload and download data,
		as well as potentially speedup it's transfer speed; 
		you will have to overload it if your class contains something special.
*/
class Stacktor{
public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Construction/Destruction/copy-construction: **/


	/** ========================================================== **/
	/*| Regular |*/

	/** ------------------------------------ **/
	// Simple constructor (constructs empty)
	__device__ __host__ inline Stacktor();
	// Destructor
	__device__ __host__ inline ~Stacktor();

	/** ------------------------------------ **/
	template<typename... Elems>
	// Constructs and fills with given elements
	__device__ __host__ inline Stacktor(const Type &elem, const Elems&... elems);

	/** ------------------------------------ **/
	template<unsigned int arrSize>
	// Constructs and copies the given fixed size array to the Stacktor
	__device__ __host__ inline Stacktor(const Type(&arr)[arrSize]);
	template<unsigned int arrSize>
	// Makes the Stacktor equivelent of the given fixed size array
	__device__ __host__ inline Stacktor& operator=(const Type(&arr)[arrSize]);

	/** ========================================================== **/
	/*| Raw data |*/
	// Constructor for raw data (note: never use this on anything allocated on stack, or using new)
	__device__ __host__ inline void initRaw();
	// Destructor for raw data (note: never use this on anything allocated on stack, or using new)
	__device__ __host__ inline void disposeRaw();


	/** ========================================================== **/
	/*| Copy-construction |*/
	// Copy-constructor
	__device__ __host__ inline Stacktor(const Stacktor &s);
	// Operator = (copies the data)
	__device__ __host__ inline Stacktor& operator=(const Stacktor &s);
	// Pass-constructor
	__device__ __host__ inline Stacktor(Stacktor &&s);
	// Pass-assignment
	__device__ __host__ inline Stacktor& operator=(Stacktor &&s);
	// Swaps the content with the given Stacktor
	__device__ __host__ inline void swapWith(Stacktor &s);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Data manipulation: **/


	/** ========================================================== **/
	/*| Insert/remove/clear |*/

	/** ------------------------------------ **/
	// Adds the given element to the Stacktor
	__device__ __host__ inline void push(const Type &elem);
	template<typename... Rest>
	// Adds the given elements to the Stacktor (might be slightly slower)
	__device__ __host__ inline void push(const Type &elem, const Rest&... rest);
	// Adds given number of elements to the stacktor
	__device__ __host__ inline void flush(int count);
	template<unsigned int arrSize>
	// Adds the elements from the given fixed size array to the Stacktor
	__device__ __host__ inline Stacktor& operator+=(const Type(&arr)[arrSize]);

	/** ------------------------------------ **/
	// Removes the top element (the last that was added) and returns it
	__device__ __host__ inline const Type& pop();
	// Swaps the element with the given index with the last one, then pops and returns it.
	__device__ __host__ inline const Type& swapPop(int index);
	// Removes the element with the given index and returns it's value
	__device__ __host__ inline const Type& remove(int index);
	// Clears the Stacktor
	__device__ __host__ inline void clear();


	/** ========================================================== **/
	/*| Get/Set |*/

	/** ------------------------------------ **/
	// i'th element
	__device__ __host__ inline Type& operator[](int i);
	// i'th element (constant)
	__device__ __host__ inline const Type& operator[](int i)const;
	// Pointer to the i'th element
	__device__ __host__ inline Type* operator+(int i);
	// Pointer to the i'th element (constant)
	__device__ __host__ inline const Type* operator+(int i)const;

	/** ------------------------------------ **/
	// The last element
	__device__ __host__ inline Type& top();
	// The last element (constant)
	__device__ __host__ inline const Type& peek()const;

	/** ------------------------------------ **/
	// Length of the Stacktor
	__device__ __host__ inline int size()const;
	// Tells, if Stacktor is empty
	__device__ __host__ inline bool empty()const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Upload/download: **/

	
	/** ========================================================== **/
	/*| Units |*/
	
	/*
	Uploads the Stacktor to GPU
	Return value:
	.	address of the clone (NULL, in case of a failure)
	Notes:
	.	This will create no "bondage" between the source and the clone;
	*/
	inline Stacktor* upload()const;

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
	inline bool uploadAt(Stacktor *destination)const;


	/** ========================================================== **/
	/*| Arrays |*/

	/** ------------------------------------ **/

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
	inline static bool upload(const Stacktor *source, Stacktor *target, int count = 1);

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
	inline static Stacktor* upload(const Stacktor *source, int count = 1);

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
	inline static bool dispose(Stacktor *arr, int count = 1);





private:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Variables: **/
	Type *data;
	int used, allocated;
	Type stackData[localCapacity];
	struct{
		void *address;
		int size, capacity;
	}clone;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Private helpers: **/


	/** ========================================================== **/
	/*| Construct/Destruct |*/
	__device__ __host__ inline void init(bool raw);
	__device__ __host__ inline void dispose(bool raw);


	/** ========================================================== **/
	/*| Capacity |*/
	__device__ __host__ inline void demandCapacity(int newCapacity);
	__device__ __host__ inline void changeData(Type *newData, int newCapacity);


	/** ========================================================== **/
	/*| Upload/download/dispose |*/

	/** ------------------------------------ **/
	inline static bool upload(const Stacktor *source, Stacktor *hosClone, Stacktor *devTarget, int count, bool dataOnly);
	inline static void initHosClones(Stacktor *hosClone, Stacktor *devTarget, int count);
	inline static int getExternalCapacity(const Stacktor *source, int count);
	inline static bool prepDataForCpyLoad(const Stacktor *source, Stacktor *hosClone, Stacktor *devTarget, int count, Type *exHosAlloc, Type *exAlloc);
	inline static void undoDataLoadPrep(const Stacktor *source, Stacktor *hosClone, Stacktor *devTarget, int count, Type *exHosAlloc, Type *exAlloc);
	inline static bool cpyLoad(Stacktor *hosClone, Stacktor *devTarget, int count, Type *exHosAlloc, Type *exAlloc, int exCount, bool loadAll);

	/** ------------------------------------ **/
	inline static bool disposeOfUnderlyingData(Stacktor *arr, Stacktor *hosClone, int count, bool hasExternalAllocation, cudaStream_t stream);
	inline static bool disposeOfExternalData(Stacktor *arr, Stacktor *hosClone, int count, cudaStream_t stream);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Friends: **/
	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(Stacktor);
};

typedef Stacktor<char, 32> String;





#include"Stacktor.impl.cuh"
