#include"FrameBufferManager.cuh"



inline FrameBufferManager::FrameBufferManager() : handler(frameBuffer) {
	const int count = handler.gpuCount();
	if (count > 0) {
		info = new int[count];
		for (int i = 0; i < count; i++) info[i] = 0;
	}
	else info = NULL;
}
inline FrameBufferManager::~FrameBufferManager() {
	if (info != NULL) delete[] info;
}

inline FrameBuffer* FrameBufferManager::gpuHandle(int index, bool blockedAlready) {
	const int count = handler.gpuCount();
	if (count > 0 && index >= 0 && index < count) {
		FrameBuffer *rv = handler.getHandleGPU(index);
		if (((info[index] & DIRTY) != 0) || rv == NULL) {
			if (!blockedAlready) lock.lock();
			handler.uploadToGPU(index, ((info[index] & DIRTY) != 0));
			if (!blockedAlready) lock.unlock();
			rv = handler.getHandleGPU(index);
		}
		return rv;
	}
	else return NULL;
}
inline const FrameBuffer* FrameBufferManager::gpuHandle(int index)const {
	const int count = handler.gpuCount();
	if (count > 0 && index >= 0 && index < count)
		return handler.getHandleGPU(index);
	else return NULL;
}

inline FrameBuffer* FrameBufferManager::cpuHandle() {
	return (&frameBuffer);
}
inline const FrameBuffer* FrameBufferManager::cpuHandle()const {
	return (&frameBuffer);
}

inline void FrameBufferManager::edit(EditFunction editFunction, bool blockedAlready) {
	if (!blockedAlready) lock.lock();
	editFunction(frameBuffer);
	if (info != NULL) {
		int count = handler.gpuCount();
		for (int i = 0; count; i++) info[i] |= DIRTY;
	}
	if (!blockedAlready) lock.unlock();
}
inline void FrameBufferManager::lockEdit() {
	lock.lock();
}
inline void FrameBufferManager::unlockEdit() {
	lock.unlock();
}

