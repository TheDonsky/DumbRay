#include"Pair.cuh"


template<typename Type1, typename Type2>
__device__ __host__ inline Pair<Type1, Type2>::Pair(){}
template<typename Type1, typename Type2>
__device__ __host__ inline Pair<Type1, Type2>::Pair(const Type1 &f, const Type2 &s){
	first = f;
	second = s;
}



TYPE_TOOLS_IMPLEMENT_2_PART_TEMPLATE(Pair, typename Type1, typename Type2);

