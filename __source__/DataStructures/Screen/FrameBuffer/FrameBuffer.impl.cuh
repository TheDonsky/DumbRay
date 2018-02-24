#include"FrameBuffer.cuh"


#pragma once
#include"../../GeneralPurpose/Generic/Generic.cuh"
#include"../../Primitives/Pure/Color/Color.h"


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__device__ __host__ inline FrameBufferFunctionPack::FrameBufferFunctionPack() { clean(); }
__device__ __host__ inline void FrameBufferFunctionPack::clean() {
	getSizeFn = NULL;
	getColorFn = NULL;
	setColorFn = NULL;
	blendColorFn = NULL;

	getBlockSizeFn = NULL;
	getBlockCountFn = NULL;
	blockPixelLocationFn = NULL;
	getBlockPixelColorFn = NULL;
	setBlockPixelColorFn = NULL;
	blendBlockPixelColorFn = NULL;

	setResolutionFn = NULL;
	requiresBlockUpdateFn = NULL;
	updateDeviceInstanceFn = NULL;
	updateBlocksFn = NULL;
}
template<typename BufferType>
__device__ __host__ inline void FrameBufferFunctionPack::use() {
	getSizeFn = getSizeGeneric<BufferType>;
	getColorFn = getColorGeneric<BufferType>;
	setColorFn = setColorGeneric<BufferType>;
	blendColorFn = blendColorGeneric<BufferType>;

	getBlockSizeFn = getBlockSizeGeneric<BufferType>;
	getBlockCountFn = getBlockCountGeneric<BufferType>;
	blockPixelLocationFn = blockPixelLocationGeneric<BufferType>;
	getBlockPixelColorFn = getBlockPixelColorGeneric<BufferType>;
	setBlockPixelColorFn = setBlockPixelColorGeneric<BufferType>;
	blendBlockPixelColorFn = blendBlockPixelColorGeneric<BufferType>;

#ifndef __CUDA_ARCH__
	setResolutionFn = setResolutionGeneric<BufferType>;
	requiresBlockUpdateFn = requiresBlockUpdateGeneric<BufferType>;
	updateDeviceInstanceFn = updateDeviceInstanceGeneric<BufferType>;
	updateBlocksFn = updateBlocksGeneric<BufferType>;
#else
	setResolutionFn = NULL;
	requiresBlockUpdateFn = NULL;
	updateDeviceInstanceFn = NULL;
	updateBlocksFn = NULL;
#endif
}

__device__ __host__ inline void FrameBufferFunctionPack::getSize(const void *buffer, int *width, int *height)const {
	getSizeFn(buffer, width, height);
}
__device__ __host__ inline Color FrameBufferFunctionPack::getColor(const void *buffer, int x, int y)const {
	return getColorFn(buffer, x, y);
}
__device__ __host__ inline void FrameBufferFunctionPack::setColor(void *buffer, int x, int y, const Color &color)const {
	setColorFn(buffer, x, y, color);
}
__device__ __host__ inline void FrameBufferFunctionPack::blendColor(void *buffer, int x, int y, const Color &color, float amount)const {
	blendColorFn(buffer, x, y, color, amount);
}

__device__ __host__ inline int FrameBufferFunctionPack::getBlockSize(const void *buffer)const {
	return getBlockSizeFn(buffer);
}
__device__ __host__ inline int FrameBufferFunctionPack::getBlockCount(const void *buffer)const {
	return getBlockCountFn(buffer);
}
__device__ __host__ inline bool FrameBufferFunctionPack::blockPixelLocation(const void *buffer, int blockId, int pixelId, int *x, int *y)const {
	return blockPixelLocationFn(buffer, blockId, pixelId, x, y);
}
__device__ __host__ inline bool FrameBufferFunctionPack::getBlockPixelColor(const void *buffer, int blockId, int pixelId, Color *color)const {
	return getBlockPixelColorFn(buffer, blockId, pixelId, color);
}
__device__ __host__ inline bool FrameBufferFunctionPack::setBlockPixelColor(void *buffer, int blockId, int pixelId, const Color &color)const {
	return setBlockPixelColorFn(buffer, blockId, pixelId, color);
}
__device__ __host__ inline bool FrameBufferFunctionPack::blendBlockPixelColor(void *buffer, int blockId, int pixelId, const Color &color, float amount)const {
	return blendBlockPixelColorFn(buffer, blockId, pixelId, color, amount);
}


inline bool FrameBufferFunctionPack::setResolution(void *buffer, int width, int height)const {
	return setResolutionFn(buffer, width, height);
}
inline bool FrameBufferFunctionPack::requiresBlockUpdate()const {
	return requiresBlockUpdateFn();
}
inline bool FrameBufferFunctionPack::updateDeviceInstance(const void *buffer, void *deviceObject)const {
	updateDeviceInstanceFn(buffer, deviceObject);
}
inline bool FrameBufferFunctionPack::updateBlocks(void *buffer, int startBlock, int endBlock, const void *deviceObject)const {
	return updateBlocksFn(buffer, startBlock, endBlock, deviceObject);
}


template<typename BufferType>
__device__ __host__ inline void FrameBufferFunctionPack::getSizeGeneric(const void *buffer, int *width, int *height) {
	((const BufferType*)buffer)->getSize(width, height);
}
template<typename BufferType>
__device__ __host__ inline Color FrameBufferFunctionPack::getColorGeneric(const void *buffer, int x, int y) {
	return ((const BufferType*)buffer)->getColor(x, y);
}
template<typename BufferType>
__device__ __host__ inline void FrameBufferFunctionPack::setColorGeneric(void *buffer, int x, int y, const Color &color) {
	((BufferType*)buffer)->setColor(x, y, color);
}
template<typename BufferType>
__device__ __host__ inline void FrameBufferFunctionPack::blendColorGeneric(void *buffer, int x, int y, const Color &color, float amount) {
	((BufferType*)buffer)->blendColor(x, y, color, amount);
}


template<typename BufferType>
__device__ __host__ inline int FrameBufferFunctionPack::getBlockSizeGeneric(const void *buffer) {
	return ((const BufferType*)buffer)->getBlockSize();
}
template<typename BufferType>
__device__ __host__ inline int FrameBufferFunctionPack::getBlockCountGeneric(const void *buffer) {
	return ((const BufferType*)buffer)->getBlockCount();
}
template<typename BufferType>
__device__ __host__ inline bool FrameBufferFunctionPack::blockPixelLocationGeneric(const void *buffer, int blockId, int pixelId, int *x, int *y) {
	return ((const BufferType*)buffer)->blockPixelLocation(blockId, pixelId, x, y);
}
template<typename BufferType>
__device__ __host__ inline bool FrameBufferFunctionPack::getBlockPixelColorGeneric(const void *buffer, int blockId, int pixelId, Color *color) {
	return ((const BufferType*)buffer)->getBlockPixelColor(blockId, pixelId, color);
}
template<typename BufferType>
__device__ __host__ inline bool FrameBufferFunctionPack::setBlockPixelColorGeneric(void *buffer, int blockId, int pixelId, const Color &color) {
	return ((BufferType*)buffer)->setBlockPixelColor(blockId, pixelId, color);
}
template<typename BufferType>
__device__ __host__ inline bool FrameBufferFunctionPack::blendBlockPixelColorGeneric(void *buffer, int blockId, int pixelId, const Color &color, float amount) {
	return ((BufferType*)buffer)->blendBlockPixelColor(blockId, pixelId, color, amount);
}


template<typename BufferType>
inline bool FrameBufferFunctionPack::setResolutionGeneric(void *buffer, int width, int height) {
	return ((BufferType*)buffer)->setResolution(width, height);
}
template<typename BufferType>
inline bool FrameBufferFunctionPack::requiresBlockUpdateGeneric() {
	return BufferType::requiresBlockUpdate();
}
template<typename BufferType>
inline bool FrameBufferFunctionPack::updateDeviceInstanceGeneric(const void *buffer, void *deviceObject) {
	return ((const BufferType*)buffer)->updateDeviceInstance((BufferType*)deviceObject);
}
template<typename BufferType>
inline bool FrameBufferFunctionPack::updateBlocksGeneric(void *buffer, int startBlock, int endBlock, const void *deviceObject) {
	return ((BufferType*)buffer)->updateBlocks(startBlock, endBlock, (const BufferType*)deviceObject);
}






/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__device__ __host__ inline void FrameBuffer::getSize(int *width, int *height)const {
	functions().getSize(object(), width, height);
}
__device__ __host__ inline Color FrameBuffer::getColor(int x, int y)const {
	return functions().getColor(object(), x, y);
}
__device__ __host__ inline void FrameBuffer::setColor(int x, int y, const Color &color) {
	functions().setColor(object(), x, y, color);
}
__device__ __host__ inline void FrameBuffer::blendColor(int x, int y, const Color &color, float amount) {
	functions().blendColor(object(), x, y, color, amount);
}

__device__ __host__ inline int FrameBuffer::getBlockSize()const {
	return functions().getBlockSize(object());
}
__device__ __host__ inline int FrameBuffer::getBlockCount()const {
	return functions().getBlockCount(object());
}
__device__ __host__ inline bool FrameBuffer::blockPixelLocation(int blockId, int pixelId, int *x, int *y)const {
	return functions().blockPixelLocation(object(), blockId, pixelId, x, y);
}
__device__ __host__ inline bool FrameBuffer::getBlockPixelColor(int blockId, int pixelId, Color *color)const {
	return functions().getBlockPixelColor(object(), blockId, pixelId, color);
}
__device__ __host__ inline bool FrameBuffer::setBlockPixelColor(int blockId, int pixelId, const Color &color) {
	return functions().setBlockPixelColor(object(), blockId, pixelId, color);
}
__device__ __host__ inline bool FrameBuffer::blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount) {
	return functions().blendBlockPixelColor(object(), blockId, pixelId, color, amount);
}


inline bool FrameBuffer::setResolution(int width, int height) {
	return functions().setResolution(object(), width, height);
}
inline bool FrameBuffer::requiresBlockUpdate()const {
	return functions().requiresBlockUpdate();
}
// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
inline bool FrameBuffer::updateDeviceInstance(void *deviceObject)const {
	char cloneMemory[sizeof(FrameBuffer)];
	FrameBuffer *clone = ((FrameBuffer*)cloneMemory);
	cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
	if (cudaMemcpyAsync(&clone, deviceObject, sizeof(FrameBuffer), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
		cudaStreamDestroy(stream); return false;
	}
	else if (cudaStreamDestroy(stream) != cudaSuccess) return false;
	else return functions().updateDeviceInstance(object(), clone->object());
}
// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
inline bool FrameBuffer::updateBlocks(int startBlock, int endBlock, const FrameBuffer *deviceObject) {
	if (functions().requiresBlockUpdate()) {
		char cloneMemory[sizeof(FrameBuffer)];
		FrameBuffer *clone = ((FrameBuffer*)cloneMemory);
		cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
		if (cudaMemcpyAsync(&clone, deviceObject, sizeof(FrameBuffer), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
			cudaStreamDestroy(stream); return false;
		}
		else if (cudaStreamDestroy(stream) != cudaSuccess) return false;
		else return functions().updateBlocks(object(), startBlock, endBlock, clone->object());
	}
	else return true;
}

inline FrameBuffer* FrameBuffer::upload()const {
	return (FrameBuffer*)(Generic<FrameBufferFunctionPack>::upload());
}
inline FrameBuffer* FrameBuffer::upload(const FrameBuffer *source, int count) {
	return (FrameBuffer*)(Generic<FrameBufferFunctionPack>::upload(source, count));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
COPY_TYPE_TOOLS_IMPLEMENTATION(FrameBuffer, Generic<FrameBufferFunctionPack>);






/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
inline FrameBuffer::BlockBank::BlockBank() {
	left = 0;
}
inline void FrameBuffer::BlockBank::reset(const FrameBuffer &buffer) {
	left = buffer.getBlockCount();
}
inline bool FrameBuffer::BlockBank::getBlocks(int count, int *start, int *end) {
	if (left <= 0) return false;
	else {
		lock.lock();
		(*end) = left;
		left -= count;
		(*start) = max(0, left);
		lock.unlock();
	}
}
