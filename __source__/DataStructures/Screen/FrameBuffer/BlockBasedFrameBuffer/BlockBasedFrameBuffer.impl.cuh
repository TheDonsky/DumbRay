#include "BlockBasedFrameBuffer.cuh"


__device__ __host__ inline BlockBasedFrameBuffer::BlockBasedFrameBuffer(int blockWidth, int blockHeight) {
	// __TODO__...
}
__device__ __host__ inline BlockBasedFrameBuffer::~BlockBasedFrameBuffer() {
	// __TODO__...
}

inline BlockBasedFrameBuffer::BlockBasedFrameBuffer(const BlockBasedFrameBuffer &other) {
	// __TODO__...
}
inline BlockBasedFrameBuffer& BlockBasedFrameBuffer::operator=(const BlockBasedFrameBuffer &other) {
	// __TODO__...
	return (*this);
}
inline bool BlockBasedFrameBuffer::copyFrom(const BlockBasedFrameBuffer &other) {
	// __TODO__...
	return true;
}

__device__ __host__ inline BlockBasedFrameBuffer::BlockBasedFrameBuffer(BlockBasedFrameBuffer &&other) {
	// __TODO__...
}
__device__ __host__ inline BlockBasedFrameBuffer& BlockBasedFrameBuffer::operator=(BlockBasedFrameBuffer &&other) {
	// __TODO__...
	return (*this);
}
__device__ __host__ inline void BlockBasedFrameBuffer::stealFrom(const BlockBasedFrameBuffer &other) {
	// __TODO__...
}

__device__ __host__ inline void BlockBasedFrameBuffer::swapWith(const BlockBasedFrameBuffer &other) {
	// __TODO__...
}


__device__ __host__ inline void BlockBasedFrameBuffer::getSize(int *width, int *height)const {
	// __TODO__...
}
__device__ __host__ inline Color BlockBasedFrameBuffer::getColor(int x, int y)const {
	// __TODO__...
	return Color(0, 0, 0, 0);
}
__device__ __host__ inline void BlockBasedFrameBuffer::setColor(int x, int y, const Color &color) {
	// __TODO__...
}
__device__ __host__ inline void BlockBasedFrameBuffer::blendColor(int x, int y, const Color &color, float amount) {
	// __TODO__...
}

__device__ __host__ inline int BlockBasedFrameBuffer::getBlockSize()const {
	// __TODO__...
	return 0;
}
__device__ __host__ inline int BlockBasedFrameBuffer::getBlockCount()const {
	// __TODO__...
	return 0;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::blockPixelLocation(int blockId, int pixelId, int *x, int *y)const {
	// __TODO__...
	return false;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::getBlockPixelColor(int blockId, int pixelId, Color *color)const {
	// __TODO__...
	return false;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::setBlockPixelColor(int blockId, int pixelId, const Color &color) {
	// __TODO__...
	return false;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount) {
	// __TODO__...
	return false;
}

inline bool BlockBasedFrameBuffer::setResolution(int width, int height) {
	// __TODO__...
	return true;
}
inline bool BlockBasedFrameBuffer::requiresBlockUpdate() {
	// __TODO__...
	return true;
}
inline bool BlockBasedFrameBuffer::updateDeviceInstance(BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream)const {
	// __TODO__...
	return true;
}
inline bool BlockBasedFrameBuffer::updateBlocks(int startBlock, int endBlock, const BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream) {
	// __TODO__...
	return true;
}

