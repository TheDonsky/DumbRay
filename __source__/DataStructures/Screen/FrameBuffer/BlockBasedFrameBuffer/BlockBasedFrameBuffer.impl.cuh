#include "BlockBasedFrameBuffer.cuh"


template<size_t blockW, size_t blockH>
__device__ __host__ inline BlockBasedFrameBuffer<blockW, blockH>::BlockBasedFrameBuffer() {
	buffer.imageW = buffer.imageH = 0;
	buffer.blockCount = 0;
	buffer.allocCount = 0;
	buffer.data = NULL;
}

template<size_t blockW, size_t blockH>
__device__ __host__ inline BlockBasedFrameBuffer<blockW, blockH>::~BlockBasedFrameBuffer() {
	clear();
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline void BlockBasedFrameBuffer<blockW, blockH>::clear() {
	if (buffer.data != NULL) {
		delete[] buffer.data; // May or may not be replaced with the mapped memory down the line;
		buffer.data = NULL;
	}
	buffer.imageW = buffer.imageH = 0;
	buffer.blockCount = 0;
	buffer.allocCount = 0;
}

template<size_t blockW, size_t blockH>
inline BlockBasedFrameBuffer<blockW, blockH>::BlockBasedFrameBuffer(const BlockBasedFrameBuffer &other)
	: BlockBasedFrameBuffer() {
	copyFrom(other);
}
template<size_t blockW, size_t blockH>
inline BlockBasedFrameBuffer<blockW, blockH>& BlockBasedFrameBuffer<blockW, blockH>::operator=(const BlockBasedFrameBuffer &other) {
	copyFrom(other);
	return (*this);
}
template<size_t blockW, size_t blockH>
inline void BlockBasedFrameBuffer<blockW, blockH>::copyFrom(const BlockBasedFrameBuffer &other) {
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

template<size_t blockW, size_t blockH>
__device__ __host__ inline BlockBasedFrameBuffer<blockW, blockH>::BlockBasedFrameBuffer(BlockBasedFrameBuffer &&other)
	: BlockBasedFrameBuffer() {
	stealFrom(other);
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline BlockBasedFrameBuffer<blockW, blockH>& BlockBasedFrameBuffer<blockW, blockH>::operator=(BlockBasedFrameBuffer &&other) {
	stealFrom(other);
	return (*this);
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline void BlockBasedFrameBuffer<blockW, blockH>::stealFrom(BlockBasedFrameBuffer &other) {
	swapWith(other);
}

template<size_t blockW, size_t blockH>
__device__ __host__ inline void BlockBasedFrameBuffer<blockW, blockH>::swapWith(BlockBasedFrameBuffer &other) {
	if (this == (&other)) return;
	TypeTools<BlockBufferData>::swap(buffer, other.buffer);
}


template<size_t blockW, size_t blockH>
__device__ __host__ inline void BlockBasedFrameBuffer<blockW, blockH>::getSize(int *width, int *height)const {
	(*width) = buffer.imageW;
	(*height) = buffer.imageH;
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline Color BlockBasedFrameBuffer<blockW, blockH>::getColor(int x, int y)const {
	int blockId, pixelId;
	if (pixelBlockLocation(x, y, &blockId, &pixelId))
		return buffer.data[((blockW * blockH) * blockId) + pixelId];
	return Color(0, 0, 0, 0);
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline void BlockBasedFrameBuffer<blockW, blockH>::setColor(int x, int y, const Color &color) {
	int blockId, pixelId;
	if (pixelBlockLocation(x, y, &blockId, &pixelId))
		buffer.data[((blockW * blockH) * blockId) + pixelId] = color;
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline void BlockBasedFrameBuffer<blockW, blockH>::blendColor(int x, int y, const Color &color, float amount) {
	int blockId, pixelId;
	if (pixelBlockLocation(x, y, &blockId, &pixelId)) {
		Color &colorRef = buffer.data[((blockW * blockH) * blockId) + pixelId];
		colorRef = ((color * amount) + (colorRef * (1.0f - amount)));
	}
}

template<size_t blockW, size_t blockH>
__device__ __host__ inline int BlockBasedFrameBuffer<blockW, blockH>::getBlockSize()const {
	return (blockW * blockH);
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline int BlockBasedFrameBuffer<blockW, blockH>::getBlockCount()const {
	return buffer.blockCount;
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline bool BlockBasedFrameBuffer<blockW, blockH>::pixelBlockLocation(int x, int y, int *blockId, int *pixelId)const {
	register int imageW = buffer.imageW;
	if (x >= imageW || y >= buffer.imageH) return false;

	register int numWidthBlocks = ((imageW + blockW - 1) / blockW);
	
	register int blockX = (x / blockW);
	register int blockY = (y / blockH);
	(*blockId) = ((blockY * numWidthBlocks) + blockX);

	register int offX = (x - (blockW * blockX));
	register int offY = (y - (blockH * blockY));
	(*pixelId) = ((offY * blockW) + offX);

	return true;
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline bool BlockBasedFrameBuffer<blockW, blockH>::blockPixelLocation(int blockId, int pixelId, int *x, int *y)const {
	if (blockId >= buffer.blockCount || pixelId >= (blockW * blockH)) return false;
	
	register int numWidthBlocks = ((buffer.imageW + blockW - 1) / blockW);
	register int blockY = (blockId / numWidthBlocks);
	register int blockX = (blockId - (blockY * numWidthBlocks));

	register int offY = (pixelId / blockW);
	register int offX = (pixelId - (offY * blockW));

	int posX = ((blockW * blockX) + offX);
	int posY = ((blockH * blockY) + offY);
	if (posX >= buffer.imageW || posY >= buffer.imageH) return false;

	(*x) = posX;
	(*y) = posY;

	return true;
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline Color BlockBasedFrameBuffer<blockW, blockH>::getBlockPixelColor(int blockId, int pixelId)const {
	return buffer.data[((blockW * blockH) * blockId) + pixelId];
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline void BlockBasedFrameBuffer<blockW, blockH>::setBlockPixelColor(int blockId, int pixelId, const Color &color) {
	buffer.data[((blockW * blockH) * blockId) + pixelId] = color;
}
template<size_t blockW, size_t blockH>
__device__ __host__ inline void BlockBasedFrameBuffer<blockW, blockH>::blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount) {
	Color &colorRef = buffer.data[((blockW * blockH) * blockId) + pixelId];
	colorRef = ((color * amount) + (colorRef * (1.0f - amount)));
}

template<size_t blockW, size_t blockH>
inline bool BlockBasedFrameBuffer<blockW, blockH>::setResolution(int width, int height) {
	if (buffer.imageW == width && buffer.imageH == height) return true;
	clear();
	buffer.imageW = width;
	buffer.imageH = height;
	buffer.blockCount = 
		(((buffer.imageW + blockW - 1) / blockW) 
			* ((buffer.imageH + blockH - 1) / blockH));
	buffer.allocCount = (buffer.blockCount * (blockW * blockH));
	buffer.data = new Color[buffer.allocCount]; // May or may not change to mapped memory...
	if (buffer.data != NULL) return true;
	clear();
	return false;
}
template<size_t blockW, size_t blockH>
inline bool BlockBasedFrameBuffer<blockW, blockH>::requiresBlockUpdate() { return true; }
template<size_t blockW, size_t blockH>
inline bool BlockBasedFrameBuffer<blockW, blockH>::updateDeviceInstance(BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream)const {
	BlockBufferData data;
	{
		std::lock_guard<std::mutex> guard(BlockBufferData::deviceReferenceLock);
		BlockBufferData::DeviceReferenceMirrors::iterator iter = BlockBufferData::deviceReferenceMirrors.find(deviceObject);
		if (iter != BlockBufferData::deviceReferenceMirrors.end()) data = iter->second;
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
template<size_t blockW, size_t blockH>
inline bool BlockBasedFrameBuffer<blockW, blockH>::updateBlocks(int startBlock, int endBlock, const BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream) {
	BlockBufferData data;
	{
		std::lock_guard<std::mutex> guard(BlockBufferData::deviceReferenceLock);
		BlockBufferData::DeviceReferenceMirrors::iterator iter = BlockBufferData::deviceReferenceMirrors.find(deviceObject);
		if (iter != BlockBufferData::deviceReferenceMirrors.end()) data = iter->second;
		else return false;
	}
	bool streamPassed = (stream != NULL);
	cudaStream_t localStream;
	if (!streamPassed) {
		if (cudaStreamCreate(&localStream) != cudaSuccess) return false;
		stream = (&localStream);
	}

	register int offset = (startBlock * (blockW * blockH));
	register size_t numBytes = (sizeof(Color) * (endBlock - startBlock) * (blockW * blockH));
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
template<size_t blockW, size_t blockH>
__device__ __host__ inline void TypeTools<BlockBasedFrameBuffer<blockW, blockH> >::init(BufferType &m) { new(&m) BufferType(); }
template<size_t blockW, size_t blockH>
__device__ __host__ inline void TypeTools<BlockBasedFrameBuffer<blockW, blockH> >::dispose(BufferType &m) { m.~BufferType(); }
template<size_t blockW, size_t blockH>
__device__ __host__ inline void TypeTools<BlockBasedFrameBuffer<blockW, blockH> >::swap(BufferType &a, BufferType &b) { a.swapWith(b); }
template<size_t blockW, size_t blockH>
__device__ __host__ inline void TypeTools<BlockBasedFrameBuffer<blockW, blockH> >::transfer(BufferType &src, BufferType &dst) { dst.stealFrom(src); }

template<size_t blockW, size_t blockH>
inline bool TypeTools<BlockBasedFrameBuffer<blockW, blockH> >::prepareForCpyLoad(
	const BufferType *source, BufferType *hosClone,
	BufferType *devClone, int count) {
	
	cudaStream_t stream;
	if (cudaStreamCreate(&stream) != cudaSuccess) return false;
	
	bool success = true;
	int i = 0;
	
	for (i = 0; i < count; i++) {
		BufferType &clone = hosClone[i];
		clone.buffer = source[i].buffer;

		size_t allocationSize = sizeof(Color) * clone.allocCount();
		if (cudaMalloc(&clone.data(), allocationSize) != cudaSuccess) {
			success = false; break; }
		if (cudaMemcpyAsync(clone.data(), source[i].data(), allocationSize, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
			cudaFree(clone.data()); success = false; break; }
	}
	
	if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;
	if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	
	if (!success) for (int j = 0; j < i; j++) cudaFree(hosClone[j].data());
	else {
		BufferType::deviceReferenceLock().lock();
		for (i = 0; i < count; i++) BufferType::deviceReferenceMirrors()[devClone + i] = hosClone[i].buffer;
		BufferType::deviceReferenceLock().unlock();
	}
	return success;
}
template<size_t blockW, size_t blockH>
inline void TypeTools<BlockBasedFrameBuffer<blockW, blockH> >::undoCpyLoadPreparations(
	const BufferType *, BufferType *hosClone, BufferType *devClone, int count) {
	BufferType::deviceReferenceLock().lock();
	for (int i = 0; i < count; i++) {
		cudaFree(hosClone[i].data());
		BufferType::deviceReferenceMirrors().erase(devClone + i);
	}
	BufferType::deviceReferenceLock().unlock();
}
template<size_t blockW, size_t blockH>
inline bool TypeTools<BlockBasedFrameBuffer<blockW, blockH> >::devArrayNeedsToBeDisposed() { return true; }

template<size_t blockW, size_t blockH>
inline bool TypeTools<BlockBasedFrameBuffer<blockW, blockH> >::disposeDevArray(BufferType *buffers, int count) {
	if (count <= 0) return true;
	
	char hosCloneLocalMemory[sizeof(BufferType)];
	char *allocation;
	size_t allocationSize = (sizeof(BufferType) * count);
	BufferType *hosClone;
	
	if (count > 1) {
		allocation = new char[allocationSize];
		hosClone = ((BufferType*)allocation);
		if (hosClone == NULL) return false;
	}
	else {
		allocation = NULL;
		hosClone = ((BufferType*)hosCloneLocalMemory);
	}

	bool success = false;
	cudaStream_t stream;
	if (cudaStreamCreate(&stream) == cudaSuccess) {
		if (cudaMemcpyAsync(hosClone, buffers, allocationSize, cudaMemcpyDeviceToHost, stream) == cudaSuccess)
			if (cudaStreamSynchronize(stream) == cudaSuccess) success = true;
		if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	}
	
	if (success) {
		BufferType::deviceReferenceLock().lock();
		for (int i = 0; i < count; i++) {
			if (cudaFree(hosClone[i].data()) != cudaSuccess) success = false;
			BufferType::deviceReferenceMirrors().erase(buffers + i);
		}
		BufferType::deviceReferenceLock().unlock();
	}

	if (allocation != NULL) delete allocation;
	return success;
}


