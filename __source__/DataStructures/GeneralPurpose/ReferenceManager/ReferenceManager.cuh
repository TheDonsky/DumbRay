#pragma once
#include "../ManagedHandler/ManagedHandler.cuh"



template<typename Object>
class ReferenceManager {
public:
	inline ReferenceManager();
	inline ~ReferenceManager();

	inline Object* gpuHandle(int index);
	//inline const Object* gpuHandle(int index)const;

	inline Object* cpuHandle();
	inline const Object* cpuHandle()const;

	typedef void(*EditFunction)(Object &instance, void *aux);
	inline void edit(EditFunction editFunction, void *aux, bool blockedAlready = false);
	template<typename Function, typename... Args>
	inline void editObject(Function function, Args&... args);
	template<typename Function, typename... Args>
	inline void editObjectLocked(Function function, Args&... args);
	inline void lockEdit();
	inline void unlockEdit();
	inline void makeDirty();




private:
	enum Flags {
		DIRTY = 1
	};
	std::mutex lock;
	int *info;
	Object object;
	ManagedHandler<Object> handler;
};






#include "ReferenceManager.impl.cuh"