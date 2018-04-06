#include "BlockBasedFrameBuffer.cuh"


__device__ __host__ inline BlockBasedFrameBuffer::BlockBasedFrameBuffer(int blockWidth, int blockHeight) {
	buffer.blockW = blockWidth;
	buffer.blockH = blockHeight;
	buffer.blockSize = (buffer.blockW * buffer.blockH);
	
	buffer.imageW = buffer.imageH = 0;
	buffer.blockCount = 0;
	buffer.allocCount = 0;
	buffer.data = NULL;

#ifndef __CUDA_ARCH__
	deviceObjectCache = new DeviceObjectCache();
#else
	deviceObjectCache = NULL;
#endif
}

__device__ __host__ inline BlockBasedFrameBuffer::~BlockBasedFrameBuffer() {
	clear();
#ifndef __CUDA_ARCH__
	delete deviceObjectCache;
#endif
}
__device__ __host__ inline void BlockBasedFrameBuffer::clear() {
#ifndef __CUDA_ARCH__
	deviceObjectCache->clean();
#endif
	if (buffer.data != NULL) {
		delete[] buffer.data; // May or may not be replaced with the mapped memory down the line;
		buffer.data = NULL;
	}
	buffer.imageW = buffer.imageH = 0;
	buffer.blockCount = 0;
	buffer.allocCount = 0;
}

inline BlockBasedFrameBuffer::BlockBasedFrameBuffer(const BlockBasedFrameBuffer &other) 
	: BlockBasedFrameBuffer(other.buffer.blockW, other.buffer.blockH) {
	copyFrom(other);
}
inline BlockBasedFrameBuffer& BlockBasedFrameBuffer::operator=(const BlockBasedFrameBuffer &other) {
	copyFrom(other);
	return (*this);
}
inline void BlockBasedFrameBuffer::copyFrom(const BlockBasedFrameBuffer &other) {
	// __TODO__...
	if (this == (&other)) return;
}

__device__ __host__ inline BlockBasedFrameBuffer::BlockBasedFrameBuffer(BlockBasedFrameBuffer &&other) 
	: BlockBasedFrameBuffer(other.buffer.blockW, other.buffer.blockH) {
	stealFrom(other);
}
__device__ __host__ inline BlockBasedFrameBuffer& BlockBasedFrameBuffer::operator=(BlockBasedFrameBuffer &&other) {
	stealFrom(other);
	return (*this);
}
__device__ __host__ inline void BlockBasedFrameBuffer::stealFrom(BlockBasedFrameBuffer &other) {
	swapWith(other);
}

__device__ __host__ inline void BlockBasedFrameBuffer::swapWith(BlockBasedFrameBuffer &other) {
	if (this == (&other)) return;
	TypeTools<BufferData>::swap(buffer, other.buffer);
#ifndef __CUDA_ARCH__
	deviceObjectCache->clean();
	other.deviceObjectCache->clean();
#endif
}


__device__ __host__ inline void BlockBasedFrameBuffer::getSize(int *width, int *height)const {
	(*width) = buffer.imageW;
	(*height) = buffer.imageH;
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
	return buffer.blockSize;
}
__device__ __host__ inline int BlockBasedFrameBuffer::getBlockCount()const {
	return buffer.blockCount;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::blockPixelLocation(int blockId, int pixelId, int *x, int *y)const {
	// __TODO__...
	if (blockId >= buffer.blockCount || pixelId >= buffer.blockSize) return false;
	return true;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::getBlockPixelColor(int blockId, int pixelId, Color *color)const {
	if (blockId >= buffer.blockCount || pixelId >= buffer.blockSize) return false;
	(*color) = buffer.data[(buffer.blockSize * blockId) + pixelId];
	return true;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::setBlockPixelColor(int blockId, int pixelId, const Color &color) {
	if (blockId >= buffer.blockCount || pixelId >= buffer.blockSize) return false;
	buffer.data[(buffer.blockSize * blockId) + pixelId] = color;
	return true;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount) {
	if (blockId >= buffer.blockCount || pixelId >= buffer.blockSize) return false;
	Color &colorRef = buffer.data[(buffer.blockSize * blockId) + pixelId];
	colorRef = ((color * amount) + (colorRef * (1.0f - amount)));
	return true;
}

inline bool BlockBasedFrameBuffer::setResolution(int width, int height) {
	if (buffer.imageW == width && buffer.imageH == height) return true;
	clear();
	buffer.imageW = width;
	buffer.imageH = height;
	buffer.blockCount = 
		(((buffer.imageW + buffer.blockW - 1) / buffer.blockW) 
			* ((buffer.imageH + buffer.blockH - 1) / buffer.blockH));
	buffer.allocCount = (buffer.blockCount * buffer.blockSize);
	buffer.data = new Color[buffer.allocCount]; // May or may not change to mapped memory...
	if (buffer.data != NULL) return true;
	clear();
	return false;
}
inline bool BlockBasedFrameBuffer::requiresBlockUpdate() { return true; }
inline bool BlockBasedFrameBuffer::updateDeviceInstance(BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream)const {
	// __TODO__...
	return true;
}
inline bool BlockBasedFrameBuffer::updateBlocks(int startBlock, int endBlock, const BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream) {
	// __TODO__...
	return true;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

template<size_t CacheSize>
inline BlockBasedFrameBuffer::ObjectCache<CacheSize>::Entry::Entry() { set(NULL); }
template<size_t CacheSize>
inline void BlockBasedFrameBuffer::ObjectCache<CacheSize>::Entry::set(const BlockBasedFrameBuffer *devObj) {
	// __TODO__...;
}
template<size_t CacheSize>
inline void BlockBasedFrameBuffer::ObjectCache<CacheSize>::clean() {
	for (size_t i = 0; i < CacheSize; i++) entries[i].set(0);
}
template<size_t CacheSize>
inline BlockBasedFrameBuffer::BufferData* BlockBasedFrameBuffer::ObjectCache<CacheSize>::hostClone(const BlockBasedFrameBuffer *devObj) {
	// __TODO__...;
	return NULL;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<>
__device__ __host__ inline void TypeTools<BlockBasedFrameBuffer>::init(BlockBasedFrameBuffer &m) { new(&m) BlockBasedFrameBuffer(); }
template<>
__device__ __host__ inline void TypeTools<BlockBasedFrameBuffer>::dispose(BlockBasedFrameBuffer &m) { m.~BlockBasedFrameBuffer(); }
template<>
__device__ __host__ inline void TypeTools<BlockBasedFrameBuffer>::swap(BlockBasedFrameBuffer &a, BlockBasedFrameBuffer &b) { a.swapWith(b); }
template<>
__device__ __host__ inline void TypeTools<BlockBasedFrameBuffer>::transfer( BlockBasedFrameBuffer &src, BlockBasedFrameBuffer &dst) { dst.stealFrom(src); }

template<>
inline bool TypeTools<BlockBasedFrameBuffer>::prepareForCpyLoad(
	const BlockBasedFrameBuffer *source, BlockBasedFrameBuffer *hosClone,
	BlockBasedFrameBuffer *, int count) {
	
	cudaStream_t stream;
	if (cudaStreamCreate(&stream) != cudaSuccess) return false;
	
	bool success = true;
	int i = 0;
	
	for (i = 0; i < count; i++) {
		BlockBasedFrameBuffer &clone = hosClone[i];
		clone.buffer = source[i].buffer;

		size_t allocationSize = sizeof(Color) * clone.buffer.allocCount;
		if (cudaMalloc(&clone.buffer.data, allocationSize) != cudaSuccess) {
			success = false; break; }
		if (cudaMemcpyAsync(clone.buffer.data, source[i].buffer.data, allocationSize, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
			cudaFree(clone.buffer.data); success = false; break; }

		hosClone[i].deviceObjectCache = NULL;
	}
	
	if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;
	if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	
	if (!success) for (int j = 0; j < i; j++) cudaFree(hosClone[j].buffer.data);
	
	return success;
}
template<>
inline void TypeTools<BlockBasedFrameBuffer>::undoCpyLoadPreparations(
	const BlockBasedFrameBuffer *, BlockBasedFrameBuffer *hosClone, BlockBasedFrameBuffer *, int count) { 
	for (int j = 0; j < count; j++) cudaFree(hosClone[j].buffer.data);
}
template<>
inline bool TypeTools<BlockBasedFrameBuffer>::devArrayNeedsToBeDisposed() { return true; }

template<>
inline bool TypeTools<BlockBasedFrameBuffer>::disposeDevArray(BlockBasedFrameBuffer *buffers, int count) {
	if (count <= 0) return true;
	
	char hosCloneLocalMemory[sizeof(BlockBasedFrameBuffer)];
	char *allocation;
	size_t allocationSize = (sizeof(BlockBasedFrameBuffer) * count);
	BlockBasedFrameBuffer *hosClone;
	
	if (count > 1) {
		allocation = new char[allocationSize];
		hosClone = ((BlockBasedFrameBuffer*)allocation);
		if (hosClone == NULL) return false;
	}
	else {
		allocation = NULL;
		hosClone = ((BlockBasedFrameBuffer*)hosCloneLocalMemory);
	}

	bool success = false;
	cudaStream_t stream;
	if (cudaStreamCreate(&stream) == cudaSuccess) {
		if (cudaMemcpyAsync(hosClone, buffers, allocationSize, cudaMemcpyDeviceToHost, stream) == cudaSuccess)
			if (cudaStreamSynchronize(stream) == cudaSuccess) success = true;
		if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	}
	
	if (success) for (int i = 0; i < count; i++)
		if (cudaFree(hosClone[i].buffer.data) != cudaSuccess) success = false;

	if (allocation != NULL) delete allocation;
	return success;
}


