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
	updateDeviceBlocksFn = NULL;
	updateHostBlocksFn = NULL;
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
	updateDeviceBlocksFn = updateDeviceBlocksGeneric<BufferType>;
	updateHostBlocksFn = updateHostBlocksGeneric<BufferType>;
#else
	setResolutionFn = NULL;
	updateDeviceBlocksFn = NULL;
	updateHostBlocksFn = NULL;
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
__device__ __host__ inline Color FrameBufferFunctionPack::getBlockPixelColor(const void *buffer, int blockId, int pixelId)const {
	return getBlockPixelColorFn(buffer, blockId, pixelId);
}
__device__ __host__ inline void FrameBufferFunctionPack::setBlockPixelColor(void *buffer, int blockId, int pixelId, const Color &color)const {
	setBlockPixelColorFn(buffer, blockId, pixelId, color);
}
__device__ __host__ inline void FrameBufferFunctionPack::blendBlockPixelColor(void *buffer, int blockId, int pixelId, const Color &color, float amount)const {
	blendBlockPixelColorFn(buffer, blockId, pixelId, color, amount);
}


inline bool FrameBufferFunctionPack::setResolution(void *buffer, int width, int height)const {
	return setResolutionFn(buffer, width, height);
}
inline bool FrameBufferFunctionPack::updateDeviceBlocks(const void *buffer, void *deviceBuffer, int startBlock, int endBlock, cudaStream_t *stream)const {
	return updateDeviceBlocksFn(buffer, deviceBuffer, startBlock, endBlock, stream);
}
inline bool FrameBufferFunctionPack::updateHostBlocks(void *buffer, const void *deviceBuffer, int startBlock, int endBlock, cudaStream_t *stream)const {
	return updateHostBlocksFn(buffer, deviceBuffer, startBlock, endBlock, stream);
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
__device__ __host__ inline Color FrameBufferFunctionPack::getBlockPixelColorGeneric(const void *buffer, int blockId, int pixelId) {
	return ((const BufferType*)buffer)->getBlockPixelColor(blockId, pixelId);
}
template<typename BufferType>
__device__ __host__ inline void FrameBufferFunctionPack::setBlockPixelColorGeneric(void *buffer, int blockId, int pixelId, const Color &color) {
	((BufferType*)buffer)->setBlockPixelColor(blockId, pixelId, color);
}
template<typename BufferType>
__device__ __host__ inline void FrameBufferFunctionPack::blendBlockPixelColorGeneric(void *buffer, int blockId, int pixelId, const Color &color, float amount) {
	((BufferType*)buffer)->blendBlockPixelColor(blockId, pixelId, color, amount);
}


template<typename BufferType>
inline bool FrameBufferFunctionPack::setResolutionGeneric(void *buffer, int width, int height) {
	return ((BufferType*)buffer)->setResolution(width, height);
}
template<typename BufferType>
inline bool FrameBufferFunctionPack::updateDeviceBlocksGeneric(const void *buffer, void *deviceBuffer, int startBlock, int endBlock, cudaStream_t *stream) {
	return ((const BufferType*)buffer)->updateDeviceBlocks((BufferType*)deviceBuffer, startBlock, endBlock, stream);
}
template<typename BufferType>
inline bool FrameBufferFunctionPack::updateHostBlocksGeneric(void *buffer, const void *deviceBuffer, int startBlock, int endBlock, cudaStream_t *stream) {
	return ((BufferType*)buffer)->updateHostBlocks((const BufferType*)deviceBuffer, startBlock, endBlock, stream);
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
__device__ __host__ inline Color FrameBuffer::getBlockPixelColor(int blockId, int pixelId)const {
	return functions().getBlockPixelColor(object(), blockId, pixelId);
}
__device__ __host__ inline void FrameBuffer::setBlockPixelColor(int blockId, int pixelId, const Color &color) {
	functions().setBlockPixelColor(object(), blockId, pixelId, color);
}
__device__ __host__ inline void FrameBuffer::blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount) {
	functions().blendBlockPixelColor(object(), blockId, pixelId, color, amount);
}


inline bool FrameBuffer::setResolution(int width, int height) {
	return functions().setResolution(object(), width, height);
}
inline void *FrameBuffer::getDeviceObject(const FrameBuffer *deviceBuffer, cudaStream_t *stream) {
	char cloneMemory[sizeof(FrameBuffer)];
	FrameBuffer *clone = ((FrameBuffer*)cloneMemory);
	cudaStream_t localStream;
	if (stream == NULL) {
		if (cudaStreamCreate(&localStream) != cudaSuccess) return NULL;
		stream = (&localStream);
	}
	if (cudaMemcpyAsync(clone, deviceBuffer, sizeof(FrameBuffer), cudaMemcpyDeviceToHost, *stream) != cudaSuccess) {
		if (stream == (&localStream)) cudaStreamDestroy(localStream); return NULL;
	}
	if (cudaStreamSynchronize(*stream) != cudaSuccess) {
		if (stream == (&localStream)) cudaStreamDestroy(localStream); return NULL;
	}
	else if (stream == (&localStream)) if (cudaStreamDestroy(localStream) != cudaSuccess) return NULL;
	return clone->object();
}
// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
inline bool FrameBuffer::updateDeviceBlocks(FrameBuffer *deviceObject, int startBlock, int endBlock, cudaStream_t *stream)const {
	void *devObject = getDeviceObject(deviceObject, stream);
	if (devObject == NULL) return false;
	return functions().updateDeviceBlocks(object(), deviceObject, startBlock, endBlock, stream);
}
// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
inline bool FrameBuffer::updateHostBlocks(const FrameBuffer *deviceObject, int startBlock, int endBlock, cudaStream_t *stream) {
	void *devObject = getDeviceObject(deviceObject, stream);
	if (devObject == NULL) return false;
	return functions().updateHostBlocks(object(), devObject, startBlock, endBlock, stream);
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
	int endValue = left.fetch_sub(count);
	if (endValue <= 0) return false;
	(*end) = endValue;
	(*start) = ((endValue > count) ? (endValue - count) : 0);
	return true;
}


inline FrameBuffer::DeviceBlockManager::DeviceBlockManager(int deviceId, Settings settings, int blocksPerSM) {

	settingFlags = settings;
	errorFlags = 0;
	statusFlags = 0;


	devId = deviceId;

	if (cudaSetDevice(devId) != cudaSuccess) {
		errorFlags |= CUDA_SET_DEVICE_FAILED;
		return;
	}
	else statusFlags |= CUDA_SET_DEVICE_OK;

	numSM = Device::multiprocessorCount();

	if (numSM < 0) {
		errorFlags |= CUDA_GET_SM_COUNT_FAILED;
		return;
	}
	else statusFlags |= CUDA_GET_SM_COUNT_OK;

	batchBlocks = (numSM * blocksPerSM);

	
	if (cudaStreamCreate(&synchStream) != cudaSuccess) {
		errorFlags |= CUDA_SYNCH_STREAM_CREATE_FAILED;
		return;
	}
	else statusFlags |= CUDA_SYNCH_STREAM_CREATE_OK;

	if (cudaStreamCreate(&renderStream) != cudaSuccess) {
		errorFlags |= CUDA_RENDER_STREAM_CREATE_FAILED;
		return;
	}
	else statusFlags |= CUDA_RENDER_STREAM_CREATE_OK;


	deviceBufferObject = NULL;
	hostBuffer = NULL;
	deviceBuffer = NULL;

	blockBank = NULL;
	lastStartBlock = 0;
	lastEndBlock = 0;
}
inline FrameBuffer::DeviceBlockManager::~DeviceBlockManager() {
	if ((statusFlags & CUDA_SYNCH_STREAM_CREATE_OK) != 0) cudaStreamDestroy(synchStream);
	if ((statusFlags & CUDA_RENDER_STREAM_CREATE_OK) != 0) cudaStreamDestroy(renderStream);
}

inline FrameBuffer::DeviceBlockManager::Settings FrameBuffer::DeviceBlockManager::settings()const { return settingFlags; }
inline FrameBuffer::DeviceBlockManager::Errors FrameBuffer::DeviceBlockManager::errors()const { return errorFlags; }
inline FrameBuffer::DeviceBlockManager::Status FrameBuffer::DeviceBlockManager::status()const { return statusFlags; }


inline bool FrameBuffer::DeviceBlockManager::setBuffers(FrameBuffer *host, FrameBuffer *device, BlockBank *bank) {
	blockBank = bank;
	lastStartBlock = 0;
	lastEndBlock = 0;

	hostBuffer = host;
	if (deviceBuffer != device) {
		deviceBuffer = device;
		if (deviceBuffer != NULL) {
			deviceBufferObject = getDeviceObject(device, &synchStream);
			if (deviceBufferObject == NULL) {
				errorFlags |= CUDA_DEVICE_BUFFER_OBJECT_OBTAIN_FAILED;
				return false;
			}
			else errorFlags &= (~CUDA_DEVICE_BUFFER_OBJECT_OBTAIN_FAILED);
		}
		else {
			deviceBufferObject = NULL;
			errorFlags &= (~CUDA_DEVICE_BUFFER_OBJECT_OBTAIN_FAILED);
		}
		if (deviceBufferObject == NULL) statusFlags &= (~CUDA_DEVICE_BUFFER_PRESENT);
		else statusFlags |= CUDA_DEVICE_BUFFER_PRESENT;
	}
	return true;
}


inline bool FrameBuffer::DeviceBlockManager::getBlocks(int &start, int &end, bool refreshDeviceBlocks) {
	if (errorFlags != 0) return false;
	if (lastStartBlock != lastEndBlock) {
		if ((settingFlags & CUDA_RENDER_STREAM_AUTO_SYNCH_ON_GET) != 0) if (!synchRenderStream()) return false;
		if (!hostBuffer->functions().updateHostBlocks(hostBuffer->object(), deviceBufferObject, lastStartBlock, lastEndBlock, &synchStream)) {
			errorFlags |= CUDA_HOST_BLOCK_UPDATE_FAILED;
			lastStartBlock = lastEndBlock;
			return false;
		}
		if ((settingFlags & CUDA_BLOCK_SYNCH_STREAM_AUTO_SYNCH_ON_GET) != 0) if (!synchBlockSynchStream()) return false;
	}
	if (blockBank->getBlocks(batchBlocks, &lastStartBlock, &lastEndBlock)) {
		if (refreshDeviceBlocks)
			if (!hostBuffer->functions().updateDeviceBlocks(hostBuffer->object(), deviceBufferObject, lastStartBlock, lastEndBlock, &synchStream)) {
				errorFlags |= CUDA_UPDATE_DEVICE_INSTANCE_FAILED;
				return false;
			}
		start = lastStartBlock;
		end = lastEndBlock;
		return true;
	}
	else return false;
}

inline cudaStream_t &FrameBuffer::DeviceBlockManager::getBlockSynchStream() { return synchStream; }
inline bool FrameBuffer::DeviceBlockManager::synchBlockSynchStream() {
	if (cudaStreamSynchronize(synchStream) == cudaSuccess) return true;
	errorFlags |= CUDA_SYNCH_STREAM_CYNCH_FAILED;
	return false;
}
inline cudaStream_t &FrameBuffer::DeviceBlockManager::getRenderStream() { return renderStream; }
inline bool FrameBuffer::DeviceBlockManager::synchRenderStream() {
	if (cudaStreamSynchronize(renderStream) == cudaSuccess) return true;
	errorFlags |= CUDA_RENDER_STREAM_SYNCH_FAILED;
	return false;
}
