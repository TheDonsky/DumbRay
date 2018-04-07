#include "BlockBasedFrameBuffer.cuh"


__device__ __host__ inline BlockBasedFrameBuffer::BlockBasedFrameBuffer(int blockWidth, int blockHeight) {
	buffer.blockW = blockWidth;
	buffer.blockH = blockHeight;
	buffer.blockSize = (buffer.blockW * buffer.blockH);
	
	buffer.imageW = buffer.imageH = 0;
	buffer.blockCount = 0;
	buffer.allocCount = 0;
	buffer.data = NULL;
}

__device__ __host__ inline BlockBasedFrameBuffer::~BlockBasedFrameBuffer() {
	clear();
}
__device__ __host__ inline void BlockBasedFrameBuffer::clear() {
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
	if (this == (&other)) return;
	buffer = other.buffer;
	buffer.data = new Color[buffer.allocCount]; // May or may not change to mapped memory...
	if (buffer.data == NULL) {
		clear();
		return;
	}
	for (int i = 0; i < buffer.allocCount; i++)
		buffer.data[i] = other.buffer.data[i];
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
}


__device__ __host__ inline void BlockBasedFrameBuffer::getSize(int *width, int *height)const {
	(*width) = buffer.imageW;
	(*height) = buffer.imageH;
}
__device__ __host__ inline Color BlockBasedFrameBuffer::getColor(int x, int y)const {
	int blockId, pixelId;
	if (pixelBlockLocation(x, y, &blockId, &pixelId))
		return buffer.data[(buffer.blockSize * blockId) + pixelId];
	return Color(0, 0, 0, 0);
}
__device__ __host__ inline void BlockBasedFrameBuffer::setColor(int x, int y, const Color &color) {
	int blockId, pixelId;
	if (pixelBlockLocation(x, y, &blockId, &pixelId))
		buffer.data[(buffer.blockSize * blockId) + pixelId] = color;
}
__device__ __host__ inline void BlockBasedFrameBuffer::blendColor(int x, int y, const Color &color, float amount) {
	int blockId, pixelId;
	if (pixelBlockLocation(x, y, &blockId, &pixelId)) {
		Color &colorRef = buffer.data[(buffer.blockSize * blockId) + pixelId];
		colorRef = ((color * amount) + (colorRef * (1.0f - amount)));
	}
}

__device__ __host__ inline int BlockBasedFrameBuffer::getBlockSize()const {
	return buffer.blockSize;
}
__device__ __host__ inline int BlockBasedFrameBuffer::getBlockCount()const {
	return buffer.blockCount;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::pixelBlockLocation(int x, int y, int *blockId, int *pixelId)const {
	register int imageW = buffer.imageW;
	if (x >= imageW || y >= buffer.imageH) return false;

	register int blockW = buffer.blockW;
	register int blockH = buffer.blockH;
	register int numWidthBlocks = ((imageW + blockW - 1) / blockW);
	
	register int blockX = (x / blockW);
	register int blockY = (y / blockH);
	(*blockId) = ((blockY * numWidthBlocks) + blockX);

	register int offX = (x - (blockW * blockX));
	register int offY = (y - (blockH * blockY));
	(*pixelId) = ((offY * blockW) + offX);

	return true;
}
__device__ __host__ inline bool BlockBasedFrameBuffer::blockPixelLocation(int blockId, int pixelId, int *x, int *y)const {
	if (blockId >= buffer.blockCount || pixelId >= buffer.blockSize) return false;
	
	register int blockW = buffer.blockW;
	register int numWidthBlocks = ((buffer.imageW + blockW - 1) / blockW);
	register int blockY = (blockId / numWidthBlocks);
	register int blockX = (blockId - (blockY * numWidthBlocks));

	register int offY = (pixelId / blockW);
	register int offX = (pixelId - (offY * blockW));

	int posX = ((blockW * blockX) + offX);
	int posY = ((buffer.blockH * blockY) + offY);
	if (posX >= buffer.imageW || posY >= buffer.imageH) return false;

	(*x) = posX;
	(*y) = posY;

	return true;
}
__device__ __host__ inline Color BlockBasedFrameBuffer::getBlockPixelColor(int blockId, int pixelId)const {
	return buffer.data[(buffer.blockSize * blockId) + pixelId];
}
__device__ __host__ inline void BlockBasedFrameBuffer::setBlockPixelColor(int blockId, int pixelId, const Color &color) {
	buffer.data[(buffer.blockSize * blockId) + pixelId] = color;
}
__device__ __host__ inline void BlockBasedFrameBuffer::blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount) {
	Color &colorRef = buffer.data[(buffer.blockSize * blockId) + pixelId];
	colorRef = ((color * amount) + (colorRef * (1.0f - amount)));
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
	BufferData data;
	{
		std::lock_guard<std::mutex> guard(deviceReferenceLock);
		DeviceReferenceMirrors::iterator iter = deviceReferenceMirrors.find(deviceObject);
		if (iter != deviceReferenceMirrors.end()) data = iter->second;
		else return false;
	}
	
	bool streamPassed = (stream != NULL);
	cudaStream_t localStream;
	if (!streamPassed) {
		if (cudaStreamCreate(&localStream) != cudaSuccess) return false;
		stream = (&localStream);
	}
	
	bool success = (cudaMemcpyAsync(data.data, buffer.data, sizeof(Color) * buffer.allocCount, cudaMemcpyHostToDevice, *stream) == cudaSuccess);
	if (cudaStreamSynchronize(*stream) != cudaSuccess) success = false;
	if (!streamPassed) if (cudaStreamDestroy(localStream) != cudaSuccess) success = false;
	
	return success;
}
inline bool BlockBasedFrameBuffer::updateBlocks(int startBlock, int endBlock, const BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream) {
	BufferData data;
	{
		std::lock_guard<std::mutex> guard(deviceReferenceLock);
		DeviceReferenceMirrors::iterator iter = deviceReferenceMirrors.find(deviceObject);
		if (iter != deviceReferenceMirrors.end()) data = iter->second;
		else return false;
	}
	bool streamPassed = (stream != NULL);
	cudaStream_t localStream;
	if (!streamPassed) {
		if (cudaStreamCreate(&localStream) != cudaSuccess) return false;
		stream = (&localStream);
	}

	register int offset = (startBlock * buffer.blockSize);
	register size_t numBytes = (sizeof(Color) * (endBlock - startBlock) * buffer.blockSize);
	bool success = (cudaMemcpyAsync(buffer.data + offset, data.data + offset, numBytes, cudaMemcpyDeviceToHost, *stream) == cudaSuccess);
	
	if (!streamPassed) {
		if (cudaStreamSynchronize(localStream) != cudaSuccess) success = false;
		if (cudaStreamDestroy(localStream) != cudaSuccess) success = false;
	}
	return success;
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
	BlockBasedFrameBuffer *devClone, int count) {
	
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
	}
	
	if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;
	if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	
	if (!success) for (int j = 0; j < i; j++) cudaFree(hosClone[j].buffer.data);
	else {
		BlockBasedFrameBuffer::deviceReferenceLock.lock();
		for (i = 0; i < count; i++) BlockBasedFrameBuffer::deviceReferenceMirrors[devClone + i] = hosClone[i].buffer;
		BlockBasedFrameBuffer::deviceReferenceLock.unlock();
	}
	return success;
}
template<>
inline void TypeTools<BlockBasedFrameBuffer>::undoCpyLoadPreparations(
	const BlockBasedFrameBuffer *, BlockBasedFrameBuffer *hosClone, BlockBasedFrameBuffer *devClone, int count) {
	BlockBasedFrameBuffer::deviceReferenceLock.lock();
	for (int i = 0; i < count; i++) {
		cudaFree(hosClone[i].buffer.data);
		BlockBasedFrameBuffer::deviceReferenceMirrors.erase(devClone + i);
	}
	BlockBasedFrameBuffer::deviceReferenceLock.unlock();
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
	
	if (success) {
		BlockBasedFrameBuffer::deviceReferenceLock.lock();
		for (int i = 0; i < count; i++) {
			if (cudaFree(hosClone[i].buffer.data) != cudaSuccess) success = false;
			BlockBasedFrameBuffer::deviceReferenceMirrors.erase(buffers + i);
		}
		BlockBasedFrameBuffer::deviceReferenceLock.unlock();
	}

	if (allocation != NULL) delete allocation;
	return success;
}


