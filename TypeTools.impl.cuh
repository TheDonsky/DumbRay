#include"TypeTools.cuh"



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

/** ========================================================== **/
#define TYPE_TOOLS_REDEFINE_1_PART(ClassType, TypeName, ...) // Insert template typename list in __VA_ARGS__ (...) (Should be ignored unless you know the implementation);
#define TYPE_TOOLS_REDEFINE_1_PART_TEMPLATE(ClassType, TypeName, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_REDEFINE_2_PART(ClassType, TypeName0, TypeName1, ...) // Insert template typename list in __VA_ARGS__ (...) (Should be ignored unless you know the implementation);
#define TYPE_TOOLS_REDEFINE_2_PART_TEMPLATE(ClassType, TypeName0, TypeName1, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_REDEFINE_3_PART(ClassType, TypeName0, TypeName1, TypeName2, ...) // Insert template typename list in __VA_ARGS__ (...) (Should be ignored unless you know the implementation);
#define TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE(ClassType, TypeName0, TypeName1, TypeName2, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_REDEFINE_4_PART(ClassType, TypeName0, TypeName1, TypeName2, TypeName3, ...) // Insert template typename list in __VA_ARGS__ (...) (Should be ignored unless you know the implementation);
#define TYPE_TOOLS_REDEFINE_4_PART_TEMPLATE(ClassType, TypeName0, TypeName1, TypeName2, TypeName3, ...) // Insert template typename list in __VA_ARGS__ (...);

/** ========================================================== **/
#define TYPE_TOOLS_DEFINE_PART_TYPE(TypeName)
#define TYPE_TOOLS_DEFINE_PART_TYPES_2(TypeName0, TypeName1)
#define TYPE_TOOLS_DEFINE_PART_TYPES_3(TypeName0, TypeName1, TypeName2)
#define TYPE_TOOLS_DEFINE_PART_TYPES_4(TypeName0, TypeName1, TypeName2, TypeName3)

/** ========================================================== **/
#define TYPE_TOOLS_ADD_COMPONENT_GETTER(ClassName, VariableName0)
#define TYPE_TOOLS_ADD_COMPONENT_GETTERS_2(ClassName, VariableName0, VariableName1)
#define TYPE_TOOLS_ADD_COMPONENT_GETTERS_3(ClassName, VariableName0, VariableName1, VariableName2)
#define TYPE_TOOLS_ADD_COMPONENT_GETTERS_4(ClassName, VariableName0, VariableName1, VariableName2, VariableName3)

/** ========================================================== **/
#define TYPE_TOOLS_IMPLEMENT_1_PART(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...) (Should be ignored unless you know the implementation);
#define TYPE_TOOLS_IMPLEMENT_1_PART_TEMPLATE(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_IMPLEMENT_2_PART(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...) (Should be ignored unless you know the implementation);
#define TYPE_TOOLS_IMPLEMENT_2_PART_TEMPLATE(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_IMPLEMENT_3_PART(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...) (Should be ignored unless you know the implementation);
#define TYPE_TOOLS_IMPLEMENT_3_PART_TEMPLATE(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_IMPLEMENT_4_PART(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...) (Should be ignored unless you know the implementation);
#define TYPE_TOOLS_IMPLEMENT_4_PART_TEMPLATE(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...);





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
// DON'T BOTHER READING BELOW.....










/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

/** ========================================================== **/
#undef TYPE_TOOLS_DEFINE_PART_TYPE
#define TYPE_TOOLS_DEFINE_PART_TYPE(TypeName) \
	typedef TypeName PartType0
#undef TYPE_TOOLS_DEFINE_PART_TYPES_2
#define TYPE_TOOLS_DEFINE_PART_TYPES_2(TypeName0, TypeName1) \
	TYPE_TOOLS_DEFINE_PART_TYPE(TypeName0); \
	typedef TypeName1 PartType1
#undef TYPE_TOOLS_DEFINE_PART_TYPES_3
#define TYPE_TOOLS_DEFINE_PART_TYPES_3(TypeName0, TypeName1, TypeName2) \
	TYPE_TOOLS_DEFINE_PART_TYPES_2(TypeName0, TypeName1); \
	typedef TypeName2 PartType2
#undef TYPE_TOOLS_DEFINE_PART_TYPES_4
#define TYPE_TOOLS_DEFINE_PART_TYPES_4(TypeName0, TypeName1, TypeName2, TypeName3) \
	TYPE_TOOLS_DEFINE_PART_TYPES_3(TypeName0, TypeName1, TypeName2); \
	typedef TypeName3 PartType3

/** ========================================================== **/
#undef TYPE_TOOLS_REDEFINE_1_PART
#define TYPE_TOOLS_REDEFINE_1_PART(ClassType, TypeName, ...) \
	template<__VA_ARGS__> class TypeTools<ClassType> { \
	public: \
		typedef ClassType MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPE(TypeName); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_1_PART_TEMPLATE
#define TYPE_TOOLS_REDEFINE_1_PART_TEMPLATE(ClassType, TypeName, ...) \
	TYPE_TOOLS_REDEFINE_1_PART(ClassType<__VA_ARGS__>, TypeName, __VA_ARGS__)
#undef TYPE_TOOLS_REDEFINE_2_PART
#define TYPE_TOOLS_REDEFINE_2_PART(ClassType, TypeName0, TypeName1, ...) \
	template<__VA_ARGS__> class TypeTools<ClassType> { \
	public: \
		typedef ClassType MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPES_2(TypeName0, TypeName1); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_2_PART_TEMPLATE
#define TYPE_TOOLS_REDEFINE_2_PART_TEMPLATE(ClassType, TypeName0, TypeName1, ...) \
	TYPE_TOOLS_REDEFINE_2_PART(ClassType<__VA_ARGS__>, TypeName0, TypeName1, __VA_ARGS__)
#undef TYPE_TOOLS_REDEFINE_3_PART
#define TYPE_TOOLS_REDEFINE_3_PART(ClassType, TypeName0, TypeName1, TypeName2, ...) \
	template<__VA_ARGS__> class TypeTools<ClassType> { \
	public: \
		typedef ClassType MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPES_3(TypeName0, TypeName1, TypeName2); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE
#define TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE(ClassType, TypeName0, TypeName1, TypeName2, ...) \
	TYPE_TOOLS_REDEFINE_3_PART(ClassType<__VA_ARGS__>, TypeName0, TypeName1, TypeName2, __VA_ARGS__)
#undef TYPE_TOOLS_REDEFINE_4_PART
#define TYPE_TOOLS_REDEFINE_4_PART(ClassType, TypeName0, TypeName1, TypeName2, TypeName3, ...) \
	template<__VA_ARGS__> class TypeTools<ClassType> { \
	public: \
		typedef ClassType MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPES_4(TypeName0, TypeName1, TypeName2, TypeName3); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_4_PART_TEMPLATE
#define TYPE_TOOLS_REDEFINE_4_PART_TEMPLATE(ClassType, TypeName0, TypeName1, TypeName2, TypeName3, ...) \
	TYPE_TOOLS_REDEFINE_4_PART(ClassType<__VA_ARGS__>, TypeName0, TypeName1, TypeName2, TypeName3, __VA_ARGS__)

/** ========================================================== **/
#define TYPE_TOOLS_ADD_GENERIC_COMPONENT_GETTER(VariableType, FunctionName, VariableName) \
	__device__ __host__ inline VariableType & FunctionName() { return VariableName; } \
	__device__ __host__ inline const VariableType & FunctionName() const { return VariableName; }
#undef TYPE_TOOLS_ADD_COMPONENT_GETTER
#define TYPE_TOOLS_ADD_COMPONENT_GETTER(ClassName, VariableName0) \
	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(ClassName); \
	TYPE_TOOLS_ADD_GENERIC_COMPONENT_GETTER(TypeTools<ClassName>::PartType0, component0, VariableName0)
#undef TYPE_TOOLS_ADD_COMPONENT_GETTERS_2
#define TYPE_TOOLS_ADD_COMPONENT_GETTERS_2(ClassName, VariableName0, VariableName1) \
	TYPE_TOOLS_ADD_COMPONENT_GETTER(ClassName, VariableName0); \
	TYPE_TOOLS_ADD_GENERIC_COMPONENT_GETTER(TypeTools<ClassName>::PartType1, component1, VariableName1)
#undef TYPE_TOOLS_ADD_COMPONENT_GETTERS_3
#define TYPE_TOOLS_ADD_COMPONENT_GETTERS_3(ClassName, VariableName0, VariableName1, VariableName2) \
	TYPE_TOOLS_ADD_COMPONENT_GETTERS_2(ClassName, VariableName0, VariableName1); \
	TYPE_TOOLS_ADD_GENERIC_COMPONENT_GETTER(TypeTools<ClassName>::PartType2, component2, VariableName2)
#undef TYPE_TOOLS_ADD_COMPONENT_GETTERS_4
#define TYPE_TOOLS_ADD_COMPONENT_GETTERS_4(ClassName, VariableName0, VariableName1, VariableName2, VariableName3) \
	TYPE_TOOLS_ADD_COMPONENT_GETTERS_3(ClassName, VariableName0, VariableName1, VariableName2); \
	TYPE_TOOLS_ADD_GENERIC_COMPONENT_GETTER(TypeTools<ClassName>::PartType3, component3, VariableName3)





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
#define TYPE_TOOLS_INIT_CALL_0(name) \
	TypeTools<PartType0>::init(name.component0())
#define TYPE_TOOLS_INIT_CALL_1(name) \
	TypeTools<PartType1>::init(name.component1())
#define TYPE_TOOLS_INIT_CALL_2(name) \
	TypeTools<PartType2>::init(name.component2())
#define TYPE_TOOLS_INIT_CALL_3(name) \
	TypeTools<PartType3>::init(name.component3())

#define TYPE_TOOLS_DISPOSE_CALL_0(name) \
	TypeTools<PartType0>::dispose(name.component0())
#define TYPE_TOOLS_DISPOSE_CALL_1(name) \
	TypeTools<PartType1>::dispose(name.component1())
#define TYPE_TOOLS_DISPOSE_CALL_2(name) \
	TypeTools<PartType2>::dispose(name.component2())
#define TYPE_TOOLS_DISPOSE_CALL_3(name) \
	TypeTools<PartType3>::dispose(name.component3())

#define TYPE_TOOLS_SWAP_CALL_0(first, second) \
	TypeTools<PartType0>::swap(first.component0(), second.component0())
#define TYPE_TOOLS_SWAP_CALL_1(first, second) \
	TypeTools<PartType1>::swap(first.component1(), second.component1())
#define TYPE_TOOLS_SWAP_CALL_2(first, second) \
	TypeTools<PartType2>::swap(first.component2(), second.component2())
#define TYPE_TOOLS_SWAP_CALL_3(first, second) \
	TypeTools<PartType3>::swap(first.component3(), second.component3())

#define TYPE_TOOLS_TRANSFER_CALL_0(first, second) \
	TypeTools<PartType0>::transfer(first.component0(), second.component0())
#define TYPE_TOOLS_TRANSFER_CALL_1(first, second) \
	TypeTools<PartType1>::transfer(first.component1(), second.component1())
#define TYPE_TOOLS_TRANSFER_CALL_2(first, second) \
	TypeTools<PartType2>::transfer(first.component2(), second.component2())
#define TYPE_TOOLS_TRANSFER_CALL_3(first, second) \
	TypeTools<PartType3>::transfer(first.component3(), second.component3())

#define TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_0 \
	TypeTools<PartType0>::prepareForCpyLoad(&(source + i)->component0(), &(hosClone + i)->component0(), &(devTarget + i)->component0(), 1)
#define TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_1 \
	TypeTools<PartType1>::prepareForCpyLoad(&(source + i)->component1(), &(hosClone + i)->component1(), &(devTarget + i)->component1(), 1)
#define TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_2 \
	TypeTools<PartType2>::prepareForCpyLoad(&(source + i)->component2(), &(hosClone + i)->component2(), &(devTarget + i)->component2(), 1)
#define TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_3 \
	TypeTools<PartType3>::prepareForCpyLoad(&(source + i)->component3(), &(hosClone + i)->component3(), &(devTarget + i)->component3(), 1)

#define TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0 \
	TypeTools<PartType0>::prepareForCpyLoad(&(source + i)->component0(), &(hosClone + i)->component0(), &(devTarget + i)->component0(), 1)
#define TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1 \
	TypeTools<PartType1>::prepareForCpyLoad(&(source + i)->component1(), &(hosClone + i)->component1(), &(devTarget + i)->component1(), 1)
#define TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_2 \
	TypeTools<PartType2>::prepareForCpyLoad(&(source + i)->component2(), &(hosClone + i)->component2(), &(devTarget + i)->component2(), 1)
#define TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_3 \
	TypeTools<PartType3>::prepareForCpyLoad(&(source + i)->component3(), &(hosClone + i)->component3(), &(devTarget + i)->component3(), 1)

#define TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_0 \
	TypeTools<PartType0>::devArrayNeedsToBeDisposed()
#define TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_1 \
	TypeTools<PartType1>::devArrayNeedsToBeDisposed()
#define TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_2 \
	TypeTools<PartType2>::devArrayNeedsToBeDisposed()
#define TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_3 \
	TypeTools<PartType3>::devArrayNeedsToBeDisposed()

#define TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0 \
	TypeTools<PartType0>::disposeDevArray(&(arr + i)->component0(), 1)
#define TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_1 \
	TypeTools<PartType1>::disposeDevArray(&(arr + i)->component1(), 1)
#define TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_2 \
	TypeTools<PartType2>::disposeDevArray(&(arr + i)->component2(), 1)
#define TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_3 \
	TypeTools<PartType3>::disposeDevArray(&(arr + i)->component3(), 1)



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
#undef TYPE_TOOLS_IMPLEMENT_1_PART
#define TYPE_TOOLS_IMPLEMENT_1_PART(ClassType, ...) \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::init(ClassType &variable) { \
		TYPE_TOOLS_INIT_CALL_0(variable); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::dispose(ClassType &variable) { \
		TYPE_TOOLS_DISPOSE_CALL_0(variable); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::swap(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_SWAP_CALL_0(a, b); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::transfer(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_TRANSFER_CALL_0(a, b); \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::prepareForCpyLoad(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		int i = 0; \
		for (i = 0; i < count; i++) \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_0) break;\
		if (i < count) { \
			undoCpyLoadPreparations(source, hosClone, devTarget, i); \
			return false; \
		} \
		return true; \
	} \
	__VA_ARGS__ inline void TypeTools<ClassType>::undoCpyLoadPreparations(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		for(int i = 0; i < count; i++) \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::devArrayNeedsToBeDisposed() { \
		return TypeTools<PartType0>::devArrayNeedsToBeDisposed(); \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::disposeDevArray(ClassType *arr, int count) { \
		for(int i = 0; i < count; i++) \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0) return false; \
		return true; \
	}
#undef TYPE_TOOLS_IMPLEMENT_1_PART_TEMPLATE
#define TYPE_TOOLS_IMPLEMENT_1_PART_TEMPLATE(ClassType, ...) \
	TYPE_TOOLS_IMPLEMENT_1_PART(ClassType<__VA_ARGS__>, template<__VA_ARGS__>)





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
#undef TYPE_TOOLS_IMPLEMENT_2_PART
#define TYPE_TOOLS_IMPLEMENT_2_PART(ClassType, ...) \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::init(ClassType &variable) { \
		TYPE_TOOLS_INIT_CALL_0(variable); \
		TYPE_TOOLS_INIT_CALL_1(variable); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::dispose(ClassType &variable) { \
		TYPE_TOOLS_DISPOSE_CALL_0(variable); \
		TYPE_TOOLS_DISPOSE_CALL_1(variable); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::swap(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_SWAP_CALL_0(a, b); \
		TYPE_TOOLS_SWAP_CALL_1(a, b); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::transfer(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_TRANSFER_CALL_0(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_1(a, b); \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::prepareForCpyLoad(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		int i = 0; \
		for (i = 0; i < count; i++) { \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_0) break; \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_1) { \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
				break; \
			} \
		} \
		if (i < count) { \
			undoCpyLoadPreparations(source, hosClone, devTarget, i); \
			return false; \
		} \
		return true; \
	} \
	__VA_ARGS__ inline void TypeTools<ClassType>::undoCpyLoadPreparations(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		for(int i = 0; i < count; i++) { \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1; \
		} \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::devArrayNeedsToBeDisposed() { \
		return (TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_0 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_1); \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::disposeDevArray(ClassType *arr, int count) { \
		for(int i = 0; i < count; i++) { \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0) return false; \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_1) return false; \
		} \
		return true; \
	}
#undef TYPE_TOOLS_IMPLEMENT_2_PART_TEMPLATE
#define TYPE_TOOLS_IMPLEMENT_2_PART_TEMPLATE(ClassType, ...) \
	TYPE_TOOLS_IMPLEMENT_2_PART(ClassType<__VA_ARGS__>, template<__VA_ARGS__>)







/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
#undef TYPE_TOOLS_IMPLEMENT_3_PART
#define TYPE_TOOLS_IMPLEMENT_3_PART(ClassType, ...) \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::init(ClassType &variable) { \
		TYPE_TOOLS_INIT_CALL_0(variable); \
		TYPE_TOOLS_INIT_CALL_1(variable); \
		TYPE_TOOLS_INIT_CALL_2(variable); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::dispose(ClassType &variable) { \
		TYPE_TOOLS_DISPOSE_CALL_0(variable); \
		TYPE_TOOLS_DISPOSE_CALL_1(variable); \
		TYPE_TOOLS_DISPOSE_CALL_2(variable); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::swap(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_SWAP_CALL_0(a, b); \
		TYPE_TOOLS_SWAP_CALL_1(a, b); \
		TYPE_TOOLS_SWAP_CALL_2(a, b); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::transfer(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_TRANSFER_CALL_0(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_1(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_2(a, b); \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::prepareForCpyLoad(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		int i = 0; \
		for (i = 0; i < count; i++) { \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_0) break; \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_1) { \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
				break; \
			} \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_2) { \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1; \
				break; \
			} \
		} \
		if (i < count) { \
			undoCpyLoadPreparations(source, hosClone, devTarget, i); \
			return false; \
		} \
		return true; \
	} \
	__VA_ARGS__ inline void TypeTools<ClassType>::undoCpyLoadPreparations(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		for(int i = 0; i < count; i++) { \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1; \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_2; \
		} \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::devArrayNeedsToBeDisposed() { \
		return (TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_0 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_1 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_2); \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::disposeDevArray(ClassType *arr, int count) { \
		for(int i = 0; i < count; i++) { \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0) return false; \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_1) return false; \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_2) return false; \
		} \
		return true; \
	}
#undef TYPE_TOOLS_IMPLEMENT_3_PART_TEMPLATE
#define TYPE_TOOLS_IMPLEMENT_3_PART_TEMPLATE(ClassType, ...) \
	TYPE_TOOLS_IMPLEMENT_3_PART(ClassType<__VA_ARGS__>, template<__VA_ARGS__>)





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
#undef TYPE_TOOLS_IMPLEMENT_4_PART
#define TYPE_TOOLS_IMPLEMENT_4_PART(ClassType, ...) \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::init(ClassType &variable) { \
		TYPE_TOOLS_INIT_CALL_0(variable); \
		TYPE_TOOLS_INIT_CALL_1(variable); \
		TYPE_TOOLS_INIT_CALL_2(variable); \
		TYPE_TOOLS_INIT_CALL_3(variable); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::dispose(ClassType &variable) { \
		TYPE_TOOLS_DISPOSE_CALL_0(variable); \
		TYPE_TOOLS_DISPOSE_CALL_1(variable); \
		TYPE_TOOLS_DISPOSE_CALL_2(variable); \
		TYPE_TOOLS_DISPOSE_CALL_3(variable); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::swap(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_SWAP_CALL_0(a, b); \
		TYPE_TOOLS_SWAP_CALL_1(a, b); \
		TYPE_TOOLS_SWAP_CALL_2(a, b); \
		TYPE_TOOLS_SWAP_CALL_3(a, b); \
	} \
	__VA_ARGS__ __device__ __host__ inline void TypeTools<ClassType>::transfer(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_TRANSFER_CALL_0(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_1(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_2(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_3(a, b); \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::prepareForCpyLoad(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		int i = 0; \
		for (i = 0; i < count; i++) { \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_0) break; \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_1) { \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
				break; \
			} \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_2) { \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1; \
				break; \
			} \
			if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_3) { \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1; \
				TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_2; \
				break; \
			} \
		} \
		if (i < count) { \
			undoCpyLoadPreparations(source, hosClone, devTarget, i); \
			return false; \
		} \
		return true; \
	} \
	__VA_ARGS__ inline void TypeTools<ClassType>::undoCpyLoadPreparations(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		for(int i = 0; i < count; i++) { \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1; \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_2; \
			TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_3; \
		}; \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::devArrayNeedsToBeDisposed() { \
		return (TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_0 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_1 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_2 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_3); \
	} \
	__VA_ARGS__ inline bool TypeTools<ClassType>::disposeDevArray(ClassType *arr, int count) { \
		for(int i = 0; i < count; i++) { \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0) return false; \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_1) return false; \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_2) return false; \
			if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_3) return false; \
		} \
		return true; \
	}
#undef TYPE_TOOLS_IMPLEMENT_4_PART_TEMPLATE
#define TYPE_TOOLS_IMPLEMENT_4_PART_TEMPLATE(ClassType, ...) \
	TYPE_TOOLS_IMPLEMENT_4_PART(ClassType<__VA_ARGS__>, template<__VA_ARGS__>)


