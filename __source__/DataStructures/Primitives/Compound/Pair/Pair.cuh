#pragma once

#include"../../../GeneralPurpose/TypeTools/TypeTools.cuh"

template<typename Type1, typename Type2> struct Pair;
//TYPE_TOOLS_REDEFINE_2_PART_TEMPLATE(Pair, Type1, Type2, typename Type1, typename Type2);
template<typename Type1, typename Type2> 
class TypeTools<Pair<Type1, Type2> > {
public:
	typedef Pair<Type1, Type2> MasterType;
	typedef Type1 PartType0;
	typedef Type2 PartType1;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType);
};

template<typename Type1, typename Type2>
struct Pair{
	Type1 first;
	Type2 second;

	__device__ __host__ inline Pair();
	__device__ __host__ inline Pair(const Type1 &f, const Type2 &s);





	TYPE_TOOLS_ADD_COMPONENT_GETTERS_2(Pair, first, second);
};






#include"Pair.impl.cuh"
