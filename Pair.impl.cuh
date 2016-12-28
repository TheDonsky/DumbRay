#include"Pair.cuh"


template<typename Type1, typename Type2>
__device__ __host__ inline Pair<Type1, Type2>::Pair(){}
template<typename Type1, typename Type2>
__device__ __host__ inline Pair<Type1, Type2>::Pair(const Type1 &f, const Type2 &s){
	first = f;
	second = s;
}




template<typename Type1, typename Type2>
__device__ __host__ inline void TypeTools<Pair<Type1, Type2> >::init(Pair<Type1, Type2> &p){
	TypeTools<Type1>::init(p.first);
	TypeTools<Type2>::init(p.second);
}
template<typename Type1, typename Type2>
__device__ __host__ inline void TypeTools<Pair<Type1, Type2> >::dispose(Pair<Type1, Type2> &p){
	TypeTools<Type1>::dispose(p.first);
	TypeTools<Type2>::dispose(p.second);
}
template<typename Type1, typename Type2>
__device__ __host__ inline void TypeTools<Pair<Type1, Type2> >::swap(Pair<Type1, Type2> &a, Pair<Type1, Type2> &b){
	TypeTools<Type1>::swap(a.first, b.first);
	TypeTools<Type2>::swap(a.second, b.second);
}
template<typename Type1, typename Type2>
__device__ __host__ inline void TypeTools<Pair<Type1, Type2> >::transfer(Pair<Type1, Type2> &src, Pair<Type1, Type2> &dst){
	TypeTools<Type1>::transfer(src.first, dst.first);
	TypeTools<Type2>::transfer(src.second, dst.second);
}

template<typename Type1, typename Type2>
inline bool TypeTools<Pair<Type1, Type2> >::prepareForCpyLoad(const Pair<Type1, Type2> *source, Pair<Type1, Type2> *hosClone, Pair<Type1, Type2> *devTarget, int count){
	int i = 0;
	for (i = 0; i < count; i++){
		if (!TypeTools<Type1>::prepareForCpyLoad(&source[i].first, &hosClone[i].first, &((devTarget + i)->first), 1)) break;
		if (!TypeTools<Type2>::prepareForCpyLoad(&source[i].second, &hosClone[i].second, &((devTarget + i)->second), 1)){
			TypeTools<Type1>::undoCpyLoadPreparations(&source[i].first, &hosClone[i].first, &((devTarget + i)->first), 1);
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
inline void TypeTools<Pair<Type1, Type2> >::undoCpyLoadPreparations(const Pair<Type1, Type2> *source, Pair<Type1, Type2> *hosClone, Pair<Type1, Type2> *devTarget, int count){
	for (int i = 0; i < count; i++){
		TypeTools<Type1>::undoCpyLoadPreparations(&source[i].first, &hosClone[i].first, &((devTarget + i)->first), 1);
		TypeTools<Type2>::undoCpyLoadPreparations(&source[i].second, &hosClone[i].second, &((devTarget + i)->second), 1);
	}
}
template<typename Type1, typename Type2>
inline bool TypeTools<Pair<Type1, Type2> >::devArrayNeedsToBeDisposed(){
	return(TypeTools<Type1>::devArrayNeedsToBeDisposed() || TypeTools<Type2>::devArrayNeedsToBeDisposed());
}
template<typename Type1, typename Type2>
inline bool TypeTools<Pair<Type1, Type2> >::disposeDevArray(Pair<Type1, Type2> *arr, int count){
	for (int i = 0; i < count; i++){
		if (!TypeTools<Type1>::disposeDevArray(&((arr + i)->first), 1)) return false;
		if (!TypeTools<Type2>::disposeDevArray(&((arr + i)->second), 1)) return false;
	}
	return(true);
}
