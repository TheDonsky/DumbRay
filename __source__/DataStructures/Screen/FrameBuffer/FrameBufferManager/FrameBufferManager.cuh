#pragma once
#include"../FrameBuffer.cuh"
#include"../../../GeneralPurpose/ManagedHandler/ManagedHandler.cuh"




class FrameBufferManager {
public:
	inline FrameBufferManager();
	inline ~FrameBufferManager();

	inline FrameBuffer* gpuHandle(int index, bool blockedAlready = false);
	inline const FrameBuffer* gpuHandle(int index)const;

	inline FrameBuffer* cpuHandle();
	inline const FrameBuffer* cpuHandle()const;

	typedef void(*EditFunction)(FrameBuffer &buffer, void *aux);
	inline void edit(EditFunction editFunction, void *aux, bool blockedAlready = false);
	inline void lockEdit();
	inline void unlockEdit();




private:
	enum Flags {
		DIRTY = 1
	};
	std::mutex lock;
	int *info;
	FrameBuffer frameBuffer;
	ManagedHandler<FrameBuffer> handler;
};





#include"FrameBufferManager.impl.cuh"


