#pragma once

#include"TypeTools.cuh"

template<typename Type1, typename Type2>
struct Pair;
template<typename Type1, typename Type2>
class TypeTools<Pair<Type1, Type2> >{
public:
	typedef Pair<Type1, Type2> ElementType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(ElementType);
};



template<typename Type1, typename Type2>
struct Pair{
	Type1 first;
	Type2 second;

	__device__ __host__ inline Pair();
	__device__ __host__ inline Pair(const Type1 &f, const Type2 &s);
};





#include"Pair.impl.cuh"
