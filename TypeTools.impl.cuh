#include"TypeTools.cuh"



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

/** ========================================================== **/
#define TYPE_TOOLS_REDEFINE_1_PART(ClassType, TypeName)
#define TYPE_TOOLS_REDEFINE_1_PART_TEMPLATE(ClassType, TypeName, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_REDEFINE_2_PART(ClassType, TypeName0, TypeName1)
#define TYPE_TOOLS_REDEFINE_2_PART_TEMPLATE(ClassType, TypeName0, TypeName1, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_REDEFINE_3_PART(ClassType, TypeName0, TypeName1, TypeName2)
#define TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE(ClassType, TypeName0, TypeName1, TypeName2, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_REDEFINE_4_PART(ClassType, TypeName0, TypeName1, TypeName2, TypeName3)
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
#define TYPE_TOOLS_IMPLEMENT_1_PART(ClassType)
#define TYPE_TOOLS_IMPLEMENT_1_PART_TEMPLATE(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_IMPLEMENT_2_PART(ClassType)
#define TYPE_TOOLS_IMPLEMENT_2_PART_TEMPLATE(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_IMPLEMENT_3_PART(ClassType)
#define TYPE_TOOLS_IMPLEMENT_3_PART_TEMPLATE(ClassType, ...) // Insert template typename list in __VA_ARGS__ (...);
#define TYPE_TOOLS_IMPLEMENT_4_PART(ClassType)
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
#define TYPE_TOOLS_REDEFINE_1_PART(ClassType, TypeName) \
	template<> class TypeTools<ClassType> { \
	public: \
		typedef ClassType MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPE(TypeName); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_1_PART_TEMPLATE
#define TYPE_TOOLS_REDEFINE_1_PART_TEMPLATE(ClassType, TypeName, ...) \
	template<__VA_ARGS__> class TypeTools<ClassType<__VA_ARGS__> > { \
	public: \
		typedef ClassType<__VA_ARGS__> MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPE(TypeName); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_2_PART
#define TYPE_TOOLS_REDEFINE_2_PART(ClassType, TypeName0, TypeName1) \
	template<> class TypeTools<ClassType> { \
	public: \
		typedef ClassType MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPES_2(TypeName0, TypeName1); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_2_PART_TEMPLATE
#define TYPE_TOOLS_REDEFINE_2_PART_TEMPLATE(ClassType, TypeName0, TypeName1, ...) \
	template<__VA_ARGS__> class TypeTools<ClassType<__VA_ARGS__> > { \
	public: \
		typedef ClassType<__VA_ARGS__> MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPES_2(TypeName0, TypeName1); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_3_PART
#define TYPE_TOOLS_REDEFINE_3_PART(ClassType, TypeName0, TypeName1, TypeName2) \
	template<> class TypeTools<ClassType> { \
	public: \
		typedef ClassType MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPES_3(TypeName0, TypeName1, TypeName2); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE
#define TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE(ClassType, TypeName0, TypeName1, TypeName2, ...) \
	template<__VA_ARGS__> class TypeTools<ClassType<__VA_ARGS__> > { \
	public: \
		typedef ClassType<__VA_ARGS__> MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPES_3(TypeName0, TypeName1, TypeName2); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_4_PART
#define TYPE_TOOLS_REDEFINE_4_PART(ClassType, TypeName0, TypeName1, TypeName2, TypeName3) \
	template<> class TypeTools<ClassType> { \
	public: \
		typedef ClassType MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPES_4(TypeName0, TypeName1, TypeName2, TypeName3); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}
#undef TYPE_TOOLS_REDEFINE_4_PART_TEMPLATE
#define TYPE_TOOLS_REDEFINE_4_PART_TEMPLATE(ClassType, TypeName0, TypeName1, TypeName2, TypeName3, ...) \
	template<__VA_ARGS__> class TypeTools<ClassType<__VA_ARGS__> > { \
	public: \
		typedef ClassType<__VA_ARGS__> MasterType; \
		TYPE_TOOLS_DEFINE_PART_TYPES_4(TypeName0, TypeName1, TypeName2, TypeName3); \
		DEFINE_TYPE_TOOLS_CONTENT_FOR(MasterType); \
	}

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

/** ========================================================== **/
#define TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_1_BODY \
	int i = 0; \
	for (i = 0; i < count; i++) \
		if(!TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_CALL_0) break;\
	if (i < count) { \
		undoCpyLoadPreparations(source, hosClone, devTarget, i); \
		return false; \
	} \
	return true

#define TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_1_BODY \
	for(int i = 0; i < count; i++) \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0

#define TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_1_BODY \
	return TypeTools<PartType0>::devArrayNeedsToBeDisposed()

#define TYPE_TOOLS_DISPOSE_DEV_ARRAY_1_BODY \
	for(int i = 0; i < count; i++) \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0) return false; \
	return true


/** ========================================================== **/
#undef TYPE_TOOLS_IMPLEMENT_1_PART
#define TYPE_TOOLS_IMPLEMENT_1_PART(ClassType) \
	__device__ __host__ inline void TypeTools<ClassType>::init(ClassType &variable) { \
		TYPE_TOOLS_INIT_CALL_0(variable); \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::dispose(ClassType &variable) { \
		TYPE_TOOLS_DISPOSE_CALL_0(variable); \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::swap(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_SWAP_CALL_0(a, b); \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::transfer(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_TRANSFER_CALL_0(a, b); \
	} \
	inline bool TypeTools<ClassType>::prepareForCpyLoad(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_1_BODY; \
	} \
	inline void TypeTools<ClassType>::undoCpyLoadPreparations(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_1_BODY; \
	} \
	inline bool TypeTools<ClassType>::devArrayNeedsToBeDisposed() { \
		TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_1_BODY; \
	} \
	inline bool TypeTools<ClassType>::disposeDevArray(ClassType *arr, int count) { \
		TYPE_TOOLS_DISPOSE_DEV_ARRAY_1_BODY; \
	}
#undef TYPE_TOOLS_IMPLEMENT_1_PART_TEMPLATE
#define TYPE_TOOLS_IMPLEMENT_1_PART_TEMPLATE(ClassType, ...) \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::init(ClassType<__VA_ARGS__> &variable) { \
		TYPE_TOOLS_INIT_CALL_0(variable); \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::dispose(ClassType<__VA_ARGS__> &variable) { \
		TYPE_TOOLS_DISPOSE_CALL_0(variable); \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::swap(ClassType<__VA_ARGS__> &a, ClassType<__VA_ARGS__> &b) { \
		TYPE_TOOLS_SWAP_CALL_0(a, b); \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::transfer(ClassType<__VA_ARGS__> &a, ClassType<__VA_ARGS__> &b) { \
		TYPE_TOOLS_TRANSFER_CALL_0(a, b); \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::prepareForCpyLoad(const ClassType<__VA_ARGS__> *source, ClassType<__VA_ARGS__> *hosClone, ClassType<__VA_ARGS__> *devTarget, int count) { \
		TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_1_BODY; \
	} \
	template<__VA_ARGS__> inline void TypeTools<ClassType<__VA_ARGS__> >::undoCpyLoadPreparations(const ClassType<__VA_ARGS__> *source, ClassType<__VA_ARGS__> *hosClone, ClassType<__VA_ARGS__> *devTarget, int count) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_1_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::devArrayNeedsToBeDisposed() { \
		TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_1_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::disposeDevArray(ClassType<__VA_ARGS__> *arr, int count) { \
		TYPE_TOOLS_DISPOSE_DEV_ARRAY_1_BODY; \
	}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

/** ========================================================== **/
#define TYPE_TOOLS_INIT_2_BODY \
	TYPE_TOOLS_INIT_CALL_0(variable); \
	TYPE_TOOLS_INIT_CALL_1(variable)

#define TYPE_TOOLS_DISPOSE_2_BODY \
	TYPE_TOOLS_DISPOSE_CALL_0(variable); \
	TYPE_TOOLS_DISPOSE_CALL_1(variable);

#define TYPE_TOOLS_SWAP_2_BODY \
		TYPE_TOOLS_SWAP_CALL_0(a, b); \
		TYPE_TOOLS_SWAP_CALL_1(a, b)

#define TYPE_TOOLS_TRANSFER_2_BODY \
		TYPE_TOOLS_TRANSFER_CALL_0(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_1(a, b)

#define TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_2_BODY \
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
	return true

#define TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_2_BODY \
	for(int i = 0; i < count; i++) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1; \
	}

#define TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_2_BODY \
	return (TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_0 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_1)

#define TYPE_TOOLS_DISPOSE_DEV_ARRAY_2_BODY \
	for(int i = 0; i < count; i++) { \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0) return false; \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_1) return false; \
	} \
	return true


/** ========================================================== **/
#undef TYPE_TOOLS_IMPLEMENT_2_PART
#define TYPE_TOOLS_IMPLEMENT_2_PART(ClassType) \
	__device__ __host__ inline void TypeTools<ClassType>::init(ClassType &variable) { \
		TYPE_TOOLS_INIT_2_BODY; \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::dispose(ClassType &variable) { \
		TYPE_TOOLS_DISPOSE_2_BODY; \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::swap(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_SWAP_2_BODY; \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::transfer(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_TRANSFER_2_BODY; \
	} \
	inline bool TypeTools<ClassType>::prepareForCpyLoad(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_2_BODY; \
	} \
	inline void TypeTools<ClassType>::undoCpyLoadPreparations(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_2_BODY; \
	} \
	inline bool TypeTools<ClassType>::devArrayNeedsToBeDisposed() { \
		TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_2_BODY; \
	} \
	inline bool TypeTools<ClassType>::disposeDevArray(ClassType *arr, int count) { \
		TYPE_TOOLS_DISPOSE_DEV_ARRAY_2_BODY; \
	}
#undef TYPE_TOOLS_IMPLEMENT_2_PART_TEMPLATE
#define TYPE_TOOLS_IMPLEMENT_2_PART_TEMPLATE(ClassType, ...) \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::init(ClassType<__VA_ARGS__> &variable) { \
		TYPE_TOOLS_INIT_2_BODY; \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::dispose(ClassType<__VA_ARGS__> &variable) { \
		TYPE_TOOLS_DISPOSE_2_BODY; \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::swap(ClassType<__VA_ARGS__> &a, ClassType<__VA_ARGS__> &b) { \
		TYPE_TOOLS_SWAP_2_BODY; \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::transfer(ClassType<__VA_ARGS__> &a, ClassType<__VA_ARGS__> &b) { \
		TYPE_TOOLS_TRANSFER_2_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::prepareForCpyLoad(const ClassType<__VA_ARGS__> *source, ClassType<__VA_ARGS__> *hosClone, ClassType<__VA_ARGS__> *devTarget, int count) { \
		TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_2_BODY; \
	} \
	template<__VA_ARGS__> inline void TypeTools<ClassType<__VA_ARGS__> >::undoCpyLoadPreparations(const ClassType<__VA_ARGS__> *source, ClassType<__VA_ARGS__> *hosClone, ClassType<__VA_ARGS__> *devTarget, int count) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_2_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::devArrayNeedsToBeDisposed() { \
		TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_2_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::disposeDevArray(ClassType<__VA_ARGS__> *arr, int count) { \
		TYPE_TOOLS_DISPOSE_DEV_ARRAY_2_BODY; \
	}







/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

/** ========================================================== **/
#define TYPE_TOOLS_INIT_3_BODY \
	TYPE_TOOLS_INIT_CALL_0(variable); \
	TYPE_TOOLS_INIT_CALL_1(variable); \
	TYPE_TOOLS_INIT_CALL_2(variable)

#define TYPE_TOOLS_DISPOSE_3_BODY \
	TYPE_TOOLS_DISPOSE_CALL_0(variable); \
	TYPE_TOOLS_DISPOSE_CALL_1(variable); \
	TYPE_TOOLS_DISPOSE_CALL_2(variable);

#define TYPE_TOOLS_SWAP_3_BODY \
		TYPE_TOOLS_SWAP_CALL_0(a, b); \
		TYPE_TOOLS_SWAP_CALL_1(a, b); \
		TYPE_TOOLS_SWAP_CALL_2(a, b);

#define TYPE_TOOLS_TRANSFER_3_BODY \
		TYPE_TOOLS_TRANSFER_CALL_0(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_1(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_2(a, b);

#define TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_3_BODY \
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
	return true

#define TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_3_BODY \
	for(int i = 0; i < count; i++) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1; \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_2; \
	}

#define TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_3_BODY \
	return (TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_0 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_1 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_2)

#define TYPE_TOOLS_DISPOSE_DEV_ARRAY_3_BODY \
	for(int i = 0; i < count; i++) { \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0) return false; \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_1) return false; \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_2) return false; \
	} \
	return true


/** ========================================================== **/
#undef TYPE_TOOLS_IMPLEMENT_3_PART
#define TYPE_TOOLS_IMPLEMENT_3_PART(ClassType) \
	__device__ __host__ inline void TypeTools<ClassType>::init(ClassType &variable) { \
		TYPE_TOOLS_INIT_3_BODY; \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::dispose(ClassType &variable) { \
		TYPE_TOOLS_DISPOSE_3_BODY; \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::swap(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_SWAP_3_BODY; \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::transfer(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_TRANSFER_3_BODY; \
	} \
	inline bool TypeTools<ClassType>::prepareForCpyLoad(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_3_BODY; \
	} \
	inline void TypeTools<ClassType>::undoCpyLoadPreparations(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_3_BODY; \
	} \
	inline bool TypeTools<ClassType>::devArrayNeedsToBeDisposed() { \
		TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_3_BODY; \
	} \
	inline bool TypeTools<ClassType>::disposeDevArray(ClassType *arr, int count) { \
		TYPE_TOOLS_DISPOSE_DEV_ARRAY_3_BODY; \
	}
#undef TYPE_TOOLS_IMPLEMENT_3_PART_TEMPLATE
#define TYPE_TOOLS_IMPLEMENT_3_PART_TEMPLATE(ClassType, ...) \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::init(ClassType<__VA_ARGS__> &variable) { \
		TYPE_TOOLS_INIT_3_BODY; \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::dispose(ClassType<__VA_ARGS__> &variable) { \
		TYPE_TOOLS_DISPOSE_3_BODY; \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::swap(ClassType<__VA_ARGS__> &a, ClassType<__VA_ARGS__> &b) { \
		TYPE_TOOLS_SWAP_3_BODY; \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::transfer(ClassType<__VA_ARGS__> &a, ClassType<__VA_ARGS__> &b) { \
		TYPE_TOOLS_TRANSFER_3_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::prepareForCpyLoad(const ClassType<__VA_ARGS__> *source, ClassType<__VA_ARGS__> *hosClone, ClassType<__VA_ARGS__> *devTarget, int count) { \
		TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_3_BODY; \
	} \
	template<__VA_ARGS__> inline void TypeTools<ClassType<__VA_ARGS__> >::undoCpyLoadPreparations(const ClassType<__VA_ARGS__> *source, ClassType<__VA_ARGS__> *hosClone, ClassType<__VA_ARGS__> *devTarget, int count) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_3_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::devArrayNeedsToBeDisposed() { \
		TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_3_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::disposeDevArray(ClassType<__VA_ARGS__> *arr, int count) { \
		TYPE_TOOLS_DISPOSE_DEV_ARRAY_3_BODY; \
	}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

/** ========================================================== **/
#define TYPE_TOOLS_INIT_4_BODY \
	TYPE_TOOLS_INIT_CALL_0(variable); \
	TYPE_TOOLS_INIT_CALL_1(variable); \
	TYPE_TOOLS_INIT_CALL_2(variable); \
	TYPE_TOOLS_INIT_CALL_3(variable)

#define TYPE_TOOLS_DISPOSE_4_BODY \
	TYPE_TOOLS_DISPOSE_CALL_0(variable); \
	TYPE_TOOLS_DISPOSE_CALL_1(variable); \
	TYPE_TOOLS_DISPOSE_CALL_2(variable); \
	TYPE_TOOLS_DISPOSE_CALL_3(variable);

#define TYPE_TOOLS_SWAP_4_BODY \
		TYPE_TOOLS_SWAP_CALL_0(a, b); \
		TYPE_TOOLS_SWAP_CALL_1(a, b); \
		TYPE_TOOLS_SWAP_CALL_2(a, b); \
		TYPE_TOOLS_SWAP_CALL_3(a, b);

#define TYPE_TOOLS_TRANSFER_4_BODY \
		TYPE_TOOLS_TRANSFER_CALL_0(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_1(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_2(a, b); \
		TYPE_TOOLS_TRANSFER_CALL_3(a, b);

#define TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_4_BODY \
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
	return true

#define TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_4_BODY \
	for(int i = 0; i < count; i++) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_0; \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_1; \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_2; \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_CALL_3; \
	}

#define TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_4_BODY \
	return (TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_0 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_1 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_2 || TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_CALL_3)

#define TYPE_TOOLS_DISPOSE_DEV_ARRAY_4_BODY \
	for(int i = 0; i < count; i++) { \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_0) return false; \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_1) return false; \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_2) return false; \
		if(!TYPE_TOOLS_DISPOSE_DEV_ARRAY_CALL_3) return false; \
	} \
	return true


/** ========================================================== **/
#undef TYPE_TOOLS_IMPLEMENT_4_PART
#define TYPE_TOOLS_IMPLEMENT_4_PART(ClassType) \
	__device__ __host__ inline void TypeTools<ClassType>::init(ClassType &variable) { \
		TYPE_TOOLS_INIT_4_BODY; \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::dispose(ClassType &variable) { \
		TYPE_TOOLS_DISPOSE_4_BODY; \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::swap(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_SWAP_4_BODY; \
	} \
	__device__ __host__ inline void TypeTools<ClassType>::transfer(ClassType &a, ClassType &b) { \
		TYPE_TOOLS_TRANSFER_4_BODY; \
	} \
	inline bool TypeTools<ClassType>::prepareForCpyLoad(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_4_BODY; \
	} \
	inline void TypeTools<ClassType>::undoCpyLoadPreparations(const ClassType *source, ClassType *hosClone, ClassType *devTarget, int count) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_4_BODY; \
	} \
	inline bool TypeTools<ClassType>::devArrayNeedsToBeDisposed() { \
		TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_4_BODY; \
	} \
	inline bool TypeTools<ClassType>::disposeDevArray(ClassType *arr, int count) { \
		TYPE_TOOLS_DISPOSE_DEV_ARRAY_4_BODY; \
	}
#undef TYPE_TOOLS_IMPLEMENT_4_PART_TEMPLATE
#define TYPE_TOOLS_IMPLEMENT_4_PART_TEMPLATE(ClassType, ...) \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::init(ClassType<__VA_ARGS__> &variable) { \
		TYPE_TOOLS_INIT_4_BODY; \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::dispose(ClassType<__VA_ARGS__> &variable) { \
		TYPE_TOOLS_DISPOSE_4_BODY; \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::swap(ClassType<__VA_ARGS__> &a, ClassType<__VA_ARGS__> &b) { \
		TYPE_TOOLS_SWAP_4_BODY; \
	} \
	template<__VA_ARGS__> __device__ __host__ inline void TypeTools<ClassType<__VA_ARGS__> >::transfer(ClassType<__VA_ARGS__> &a, ClassType<__VA_ARGS__> &b) { \
		TYPE_TOOLS_TRANSFER_4_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::prepareForCpyLoad(const ClassType<__VA_ARGS__> *source, ClassType<__VA_ARGS__> *hosClone, ClassType<__VA_ARGS__> *devTarget, int count) { \
		TYPE_TOOLS_PREPARE_FOR_CPY_LOAD_4_BODY; \
	} \
	template<__VA_ARGS__> inline void TypeTools<ClassType<__VA_ARGS__> >::undoCpyLoadPreparations(const ClassType<__VA_ARGS__> *source, ClassType<__VA_ARGS__> *hosClone, ClassType<__VA_ARGS__> *devTarget, int count) { \
		TYPE_TOOLS_UNDO_CPY_LOAD_PREPARATIONS_4_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::devArrayNeedsToBeDisposed() { \
		TYPE_TOOLS_DEV_ARRAY_NEEDS_TO_BE_DISPOSED_4_BODY; \
	} \
	template<__VA_ARGS__> inline bool TypeTools<ClassType<__VA_ARGS__> >::disposeDevArray(ClassType<__VA_ARGS__> *arr, int count) { \
		TYPE_TOOLS_DISPOSE_DEV_ARRAY_4_BODY; \
	}


