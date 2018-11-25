#include"Pair.cuh"


template<typename Type1, typename Type2>
__device__ __host__ inline Pair<Type1, Type2>::Pair(){}
template<typename Type1, typename Type2>
__device__ __host__ inline Pair<Type1, Type2>::Pair(const Type1 &f, const Type2 &s){
	first = f;
	second = s;
}



//TYPE_TOOLS_IMPLEMENT_2_PART_TEMPLATE(Pair, typename Type1, typename Type2);
template<typename Type1, typename Type2> __device__ __host__ inline void TypeTools<Pair<Type1, Type2> >::init(Pair<Type1, Type2> &variable) {
	TYPE_TOOLS_INIT_CALL_0(variable);
	TYPE_TOOLS_INIT_CALL_1(variable);
}
template<typename Type1, typename Type2> __device__ __host__ inline void TypeTools<Pair<Type1, Type2> >::dispose(Pair<Type1, Type2> &variable) {
	TYPE_TOOLS_DISPOSE_CALL_0(variable);
	TYPE_TOOLS_DISPOSE_CALL_1(variable);
}
template<typename Type1, typename Type2> __device__ __host__ inline void TypeTools<Pair<Type1, Type2> >::swap(Pair<Type1, Type2> &a, Pair<Type1, Type2> &b) {
	TYPE_TOOLS_SWAP_CALL_0(a, b);
	TYPE_TOOLS_SWAP_CALL_1(a, b);
}
template<typename Type1, typename Type2> __device__ __host__ inline void TypeTools<Pair<Type1, Type2> >::transfer(Pair<Type1, Type2> &a, Pair<Type1, Type2> &b) {
	TYPE_TOOLS_TRANSFER_CALL_0(a, b);
	TYPE_TOOLS_TRANSFER_CALL_1(a, b);
}
template<typename Type1, typename Type2> inline bool TypeTools<Pair<Type1, Type2> >::prepareForCpyLoad(const Pair<Type1, Type2> *source, Pair<Type1, Type2> *hosClone, Pair<Type1, Type2> *devTarget, int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_0) break;
		if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_1) {
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0;
			break;
		}
	}
	if (i < count) {
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return false;
	}
	return true;
}
template<typename Type1, typename Type2> inline void TypeTools<Pair<Type1, Type2> >::undoCpyLoadPreparations(const Pair<Type1, Type2> *source, Pair<Type1, Type2> *hosClone, Pair<Type1, Type2> *devTarget, int count) {
	for(int i = 0; i < count; i++) {
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0;
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1;
	}
}
template<typename Type1, typename Type2> inline bool TypeTools<Pair<Type1, Type2> >::devArrayNeedsToBeDisposed() {
	return (TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_0 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_1);
}
template<typename Type1, typename Type2> inline bool TypeTools<Pair<Type1, Type2> >::disposeDevArray(Pair<Type1, Type2> *arr, int count) {
	for(int i = 0; i < count; i++) {
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0) return false;
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_1) return false;
	}
	return true;
}


