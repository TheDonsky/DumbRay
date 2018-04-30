#include "ReferenceManager.cuh"


template<typename Object>
inline ReferenceManager<Object>::ReferenceManager() : handler(object) {
	const int count = handler.gpuCount();
	if (count > 0) {
		info = new int[count];
		for (int i = 0; i < count; i++) info[i] = 0;
	}
	else info = NULL;
}
template<typename Object>
inline ReferenceManager<Object>::~ReferenceManager() {
	if (info != NULL) delete[] info;
}

template<typename Object>
inline Object* ReferenceManager<Object>::gpuHandle(int index, bool blockedAlready) {
	const int count = handler.gpuCount();
	if (count > 0 && index >= 0 && index < count) {
		if (!blockedAlready) lock.lock();
		Object *rv = handler.getHandleGPU(index);
		if (((info[index] & DIRTY) != 0) || rv == NULL) {
			handler.uploadToGPU(index, ((info[index] & DIRTY) != 0));
			info[index] &= (~((int)DIRTY));
			rv = handler.getHandleGPU(index);
		}
		if (!blockedAlready) lock.unlock();
		return rv;
	}
	else return NULL;
}
/*
template<typename Object>
inline const Object* ReferenceManager<Object>::gpuHandle(int index)const {
	const int count = handler.gpuCount();
	if (count > 0 && index >= 0 && index < count)
		return handler.getHandleGPU(index);
	else return NULL;
}
//*/

template<typename Object>
inline Object* ReferenceManager<Object>::cpuHandle() {
	return (&object);
}
template<typename Object>
inline const Object* ReferenceManager<Object>::cpuHandle()const {
	return (&object);
}

template<typename Object>
inline void ReferenceManager<Object>::edit(EditFunction editFunction, void *aux, bool blockedAlready) {
	if (blockedAlready) editObjectLocked(editFunction, aux);
	else editObject(editFunction, aux);
}
template<typename Object>
template<typename Function, typename... Args>
inline void ReferenceManager<Object>::editObject(Function function, Args&... args) {
	lockEdit();
	editObjectLocked(function, args...);
	unlockEdit();
}
template<typename Object>
template<typename Function, typename... Args>
inline void ReferenceManager<Object>::editObjectLocked(Function function, Args&... args) {
	function(object, args...);
	makeDirty();
}
template<typename Object>
inline void ReferenceManager<Object>::lockEdit() {
	lock.lock();
}
template<typename Object>
inline void ReferenceManager<Object>::unlockEdit() {
	lock.unlock();
}
template<typename Object>
inline void ReferenceManager<Object>::makeDirty() {
	if (info != NULL) {
		int count = handler.gpuCount();
		for (int i = 0; i < count; i++) info[i] |= DIRTY;
	}
}
