#include"Pair.cuh"


template<typename Type1, typename Type2>
__device__ __host__ inline Pair<Type1, Type2>::Pair(){}
template<typename Type1, typename Type2>
__device__ __host__ inline Pair<Type1, Type2>::Pair(const Type1 &f, const Type2 &s){
	first = f;
	second = s;
}




template<typename Type1, typename Type2>
__device__ __host__ inline void StacktorTypeTools<Pair<Type1, Type2> >::init(Pair<Type1, Type2> &p){
	StacktorTypeTools<Type1>::init(p.first);
	StacktorTypeTools<Type2>::init(p.second);
}
template<typename Type1, typename Type2>
__device__ __host__ inline void StacktorTypeTools<Pair<Type1, Type2> >::dispose(Pair<Type1, Type2> &p){
	StacktorTypeTools<Type1>::dispose(p.first);
	StacktorTypeTools<Type2>::dispose(p.second);
}
template<typename Type1, typename Type2>
__device__ __host__ inline void StacktorTypeTools<Pair<Type1, Type2> >::swap(Pair<Type1, Type2> &a, Pair<Type1, Type2> &b){
	StacktorTypeTools<Type1>::swap(a.first, b.first);
	StacktorTypeTools<Type2>::swap(a.second, b.second);
}
template<typename Type1, typename Type2>
__device__ __host__ inline void StacktorTypeTools<Pair<Type1, Type2> >::transfer(Pair<Type1, Type2> &src, Pair<Type1, Type2> &dst){
	StacktorTypeTools<Type1>::transfer(src.first, dst.first);
	StacktorTypeTools<Type2>::transfer(src.second, dst.second);
}

template<typename Type1, typename Type2>
inline bool StacktorTypeTools<Pair<Type1, Type2> >::prepareForCpyLoad(const Pair<Type1, Type2> *source, Pair<Type1, Type2> *hosClone, Pair<Type1, Type2> *devTarget, int count){
	int i = 0;
	for (i = 0; i < count; i++){
		if (!StacktorTypeTools<Type1>::prepareForCpyLoad(&source[i].first, &hosClone[i].first, &((devTarget + i)->first), 1)) break;
		if (!StacktorTypeTools<Type2>::prepareForCpyLoad(&source[i].second, &hosClone[i].second, &((devTarget + i)->second), 1)){
			StacktorTypeTools<Type1>::undoCpyLoadPreparations(&source[i].first, &hosClone[i].first, &((devTarget + i)->first), 1);
			return false;
		}
	}
	if (i < count){
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return(false);
	}
	return(true);
}

template<typename Type1, typename Type2>
inline void StacktorTypeTools<Pair<Type1, Type2> >::undoCpyLoadPreparations(const Pair<Type1, Type2> *source, Pair<Type1, Type2> *hosClone, Pair<Type1, Type2> *devTarget, int count){
	for (int i = 0; i < count; i++){
		StacktorTypeTools<Type1>::undoCpyLoadPreparations(&source[i].first, &hosClone[i].first, &((devTarget + i)->first), 1);
		StacktorTypeTools<Type2>::undoCpyLoadPreparations(&source[i].second, &hosClone[i].second, &((devTarget + i)->second), 1);
	}
}
template<typename Type1, typename Type2>
inline bool StacktorTypeTools<Pair<Type1, Type2> >::devArrayNeedsToBeDisoposed(){
	return(StacktorTypeTools<Type1>::devArrayNeedsToBeDisoposed() || StacktorTypeTools<Type2>::devArrayNeedsToBeDisoposed());
}
template<typename Type1, typename Type2>
inline bool StacktorTypeTools<Pair<Type1, Type2> >::disposeDevArray(Pair<Type1, Type2> *arr, int count){
	for (int i = 0; i < count; i++){
		if (!StacktorTypeTools<Type1>::disposeDevArray(&((arr + i)->first), 1)) return false;
		if (!StacktorTypeTools<Type2>::disposeDevArray(&((arr + i)->second), 1)) return false;
	}
	return(true);
}
