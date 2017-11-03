#include"FrameBuffer.cuh"


#pragma once
#include"../../GeneralPurpose/Generic/Generic.cuh"
#include"../../Primitives/Pure/Color/Color.h"


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__device__ __host__ inline FrameBufferFunctionPack::FrameBufferFunctionPack() { clean(); }
__device__ __host__ inline void FrameBufferFunctionPack::clean() {
	getSizeFunction = NULL;
	getColorFunction = NULL;
	setColorFunction = NULL;
	blendColorFunction = NULL;
	getDataFunction = NULL;
	getDataFunctionConst = NULL;
	setResolutionFunction = NULL;
	requiresBlockUpdateFunction = NULL;
	updateBlocksFunction = NULL;
}
template<typename BufferType>
__device__ __host__ inline void FrameBufferFunctionPack::use() {
	getSizeFunction = getSizeGeneric<BufferType>;
	getColorFunction = getColorGeneric<BufferType>;
	setColorFunction = setColorGeneric<BufferType>;
	blendColorFunction = blendColorGeneric<BufferType>;
	getDataFunction = getDataGeneric<BufferType>;
	getDataFunctionConst = getDataGenericConst<BufferType>;
#ifndef __CUDA_ARCH__
	setResolutionFunction = setResolutionGeneric<BufferType>;
	requiresBlockUpdateFunction = requiresBlockUpdateGeneric<BufferType>;
	updateBlocksFunction = updateBlocksGeneric<BufferType>;
#else
	setResolutionFunction = NULL;
	requiresBlockUpdateFunction = NULL;
	updateBlocksFunction = NULL;
#endif
}

__device__ __host__ inline void FrameBufferFunctionPack::getSize(const void *buffer, int &width, int &height)const {
	getSizeFunction(buffer, width, height);
}
__device__ __host__ inline Color FrameBufferFunctionPack::getColor(const void *buffer, int x, int y)const {
	return getColorFunction(buffer, x, y);
}
__device__ __host__ inline void FrameBufferFunctionPack::setColor(void *buffer, int x, int y, const Color &color)const {
	setColorFunction(buffer, x, y, color);
}
__device__ __host__ inline void FrameBufferFunctionPack::blendColor(void *buffer, int x, int y, const Color &color, float amount)const {
	blendColorFunction(buffer, x, y, color, amount);
}
__device__ __host__ inline Color* FrameBufferFunctionPack::getData(void *buffer)const {
	return getDataFunction(buffer);
}
__device__ __host__ inline const Color* FrameBufferFunctionPack::getData(const void *buffer)const {
	return getDataFunctionConst(buffer);
}
inline bool FrameBufferFunctionPack::setResolution(void *buffer, int width, int height) {
	return setResolutionFunction(buffer, width, height);
}
inline bool FrameBufferFunctionPack::requiresBlockUpdate() {
	return requiresBlockUpdateFunction();
}
inline bool FrameBufferFunctionPack::updateBlocks(
	void *buffer, int startBlock, int endBlock,
	int blockWidth, int blockHeight, const void *deviceObject) {
	return updateBlocksFunction(buffer, startBlock, endBlock, 
		blockWidth, blockHeight, deviceObject);
}


template<typename BufferType>
__device__ __host__ inline void FrameBufferFunctionPack::getSizeGeneric(const void *buffer, int &width, int &height) {
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
__device__ __host__ inline Color* FrameBufferFunctionPack::getDataGeneric(void *buffer) {
	return ((BufferType*)buffer)->getData();
}
template<typename BufferType>
__device__ __host__ inline const Color* FrameBufferFunctionPack::getDataGenericConst(const void *buffer) {
	return ((const BufferType*)buffer)->getData();
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
inline bool FrameBufferFunctionPack::updateBlocksGeneric(
	void *buffer, int startBlock, int endBlock,
	int blockWidth, int blockHeight, const void *deviceObject) {
	return ((BufferType*)buffer)->updateBlocks(startBlock, endBlock,
		blockWidth, blockHeight, (const BufferType*)deviceObject);
}






/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__device__ __host__ inline void FrameBuffer::getSize(int &width, int &height)const {
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
__device__ __host__ inline Color* FrameBuffer::getData() {
	return functions().getData(object());
}
__device__ __host__ inline const Color* FrameBuffer::getData()const {
	return functions().getData(object());
}

inline bool FrameBuffer::setResolution(int width, int height) {
	return functions().setResolution(object(), width, height);
}

// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
inline bool FrameBuffer::updateBlocks(int startBlock, int endBlock,
	int blockWidth, int blockHeight, const FrameBuffer *deviceObject) {
	if (functions().requiresBlockUpdate()) {
		char clone[sizeof(FrameBuffer)];
		if (cudaMemcpy(&clone, deviceObject, sizeof(FrameBuffer), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
		else return updateBlocks(startBlock, endBlock, blockWidth, blockHeight,
			((FrameBuffer*)((void*)clone))->object());
	}
	else return true;
}
// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
inline bool FrameBuffer::updateBlocks(int startBlock, int endBlock,
	int blockWidth, int blockHeight, const void *deviceObject) {
	return functions().updateBlocks(object(), startBlock, endBlock,
		blockWidth, blockHeight, deviceObject);
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

