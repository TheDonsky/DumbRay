#pragma once

#include"Stacktor.cuh"




template<typename Type, unsigned int base = 32>
class IntMap{
public:
	__device__ __host__ inline IntMap();

	__device__ __host__ inline bool contains(int key)const;
	__device__ __host__ inline Type& operator[](int key);
	__device__ __host__ inline const Type& operator[](int key)const;
	__device__ __host__ inline void put(int key, Type value);
	__device__ __host__ inline void remove(int key);
	__device__ __host__ inline void clear();


private:
	struct Node{
		__device__ __host__ inline Node();

		Type value;
		bool containsValue;
		int children;
	};
	Stacktor<Node, 2> nodes;

	__device__ __host__ inline int forceFind(int key);
	__device__ __host__ inline int find(int key) const;
	__device__ __host__ inline int getStart(int &key) const;
};



#include"IntMap.impl.h"
