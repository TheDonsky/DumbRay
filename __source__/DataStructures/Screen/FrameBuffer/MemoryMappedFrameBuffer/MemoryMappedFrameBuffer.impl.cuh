#include"MemoryMappedFrameBuffer.cuh"


__device__ __host__ inline MemoryMappedFrameBuffer::MemoryMappedFrameBuffer(int blockWidth, int blockHeight) {
	allocSize = -1;
	sizeX = -1;
	sizeY = -1;
	blockW = blockWidth;
	blockH = blockHeight;
	data = NULL;
	flags = 0;
}
__device__ __host__ inline MemoryMappedFrameBuffer::~MemoryMappedFrameBuffer() {
	clear();
}
__device__ __host__ inline void MemoryMappedFrameBuffer::clear() {
#ifndef __CUDA_ARCH__
	if (data != NULL) {
		if ((flags & CAN_NOT_USE_DEVICE) != 0) delete data;
		else cudaFreeHost(data);
		data = NULL;
	}
#else
	data = NULL;
#endif
	allocSize = -1;
	sizeX = -1;
	sizeY = -1;
}

inline MemoryMappedFrameBuffer::MemoryMappedFrameBuffer(const MemoryMappedFrameBuffer &other) 
	: MemoryMappedFrameBuffer() {
	(*this) = other;
}
inline MemoryMappedFrameBuffer& MemoryMappedFrameBuffer::operator=(const MemoryMappedFrameBuffer &other) {
	copyFrom(other);
	return (*this);
}
inline bool MemoryMappedFrameBuffer::copyFrom(const MemoryMappedFrameBuffer &other, bool accelerate) {
	if (this == (&other)) return true;
	if (other.data == NULL) {
		clear();
		return true;
	}
	if (setResolution(other.sizeX, other.sizeY)) {
		const int size = (sizeX * sizeY);
		if (accelerate && ((flags & CAN_NOT_USE_DEVICE) == 0) && ((other.flags & CAN_NOT_USE_DEVICE) == 0))
			return (cudaMemcpy(data, other.data, sizeof(Color) * size, cudaMemcpyDefault) == cudaSuccess);
		for (int i = 0; i < size; i++) data[i] = other.data[i];
		return true;
	}
	else return false;
}

__device__ __host__ inline MemoryMappedFrameBuffer::MemoryMappedFrameBuffer(MemoryMappedFrameBuffer &&other)
	: MemoryMappedFrameBuffer() {
	stealFrom(other);
}
__device__ __host__ inline MemoryMappedFrameBuffer& MemoryMappedFrameBuffer::operator=(MemoryMappedFrameBuffer &&other) {
	stealFrom(other);
	return (*this);
}
__device__ __host__ inline void MemoryMappedFrameBuffer::stealFrom(MemoryMappedFrameBuffer &other) {
	if (this == (&other)) return;
	clear();
	allocSize = other.allocSize;
	sizeX = other.sizeX;
	sizeY = other.sizeY;
	data = other.data;
	flags = other.flags;
	other.allocSize = -1;
	other.sizeX = -1;
	other.sizeY = -1;
	other.data = NULL;
}

__device__ __host__ inline void MemoryMappedFrameBuffer::swapWith(MemoryMappedFrameBuffer &other) {
	TypeTools<int>::swap(allocSize, other.allocSize);
	TypeTools<int>::swap(sizeX, other.sizeX);
	TypeTools<int>::swap(sizeY, other.sizeY);
	TypeTools<ColorPointer>::swap(data, other.data);
	TypeTools<int>::swap(flags, other.flags);
}


inline bool MemoryMappedFrameBuffer::setResolution(int width, int height) {
	if (width <= 0 || height <= 0) return false;
	int needed = (width * height);
	if (allocSize < needed) {
		if ((flags & CAN_NOT_USE_DEVICE) != 0) {
			if (data != NULL) delete[] data;
			data = new Color[needed];
			if (data == NULL) {
				allocSize = -1;
				sizeX = -1;
				sizeY = -1;
				return false;
			}
		}
		else {
			if (data != NULL) cudaFreeHost(data);
			if (cudaHostAlloc(&data, sizeof(Color) * needed, 
				/* cudaHostAllocWriteCombined | /**/ cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
				data = NULL;
				flags |= CAN_NOT_USE_DEVICE;
				return setResolution(width, height);
			}
		}
		allocSize = needed;
	}
	sizeX = width;
	sizeY = height;
	return true;
}
inline bool MemoryMappedFrameBuffer::requiresBlockUpdate() { return false; }
inline bool MemoryMappedFrameBuffer::updateDeviceInstance(MemoryMappedFrameBuffer *deviceObject, cudaStream_t *)const { return (deviceObject != NULL); }
inline bool MemoryMappedFrameBuffer::updateBlocks(int startBlock, int endBlock, const MemoryMappedFrameBuffer *deviceObject, cudaStream_t *) {
	return ((startBlock >= 0) && (endBlock >= 0) && (deviceObject != NULL));
}

__device__ __host__ inline void MemoryMappedFrameBuffer::getSize(int *width, int *height)const {
	(*width) = sizeX;
	(*height) = sizeY;
}
__device__ __host__ inline Color MemoryMappedFrameBuffer::getColor(int x, int y)const {
	return data[(sizeX * y) + x];
}
__device__ __host__ inline void MemoryMappedFrameBuffer::setColor(int x, int y, const Color &color) {
	data[(sizeX * y) + x] = color;
}
__device__ __host__ inline void MemoryMappedFrameBuffer::blendColor(int x, int y, const Color &color, float amount) {
	register int id = ((sizeX * y) + x);
	data[id] = ((color * amount) + data[id] * (1.0f - amount));
}

__device__ __host__ inline Color* MemoryMappedFrameBuffer::getData() {
	return data;
}
__device__ __host__ inline const Color* MemoryMappedFrameBuffer::getData()const {
	return data;
}

__device__ __host__ inline int MemoryMappedFrameBuffer::blockCount(int dataSize, int blockSize) {
	return ((dataSize <= 0) ? 0 : ((dataSize + blockSize - 1) / blockSize));
}
__device__ __host__ inline int MemoryMappedFrameBuffer::widthBlocks()const {
	return blockCount(sizeX, blockW);
}
__device__ __host__ inline int MemoryMappedFrameBuffer::heightBlocks()const {
	return blockCount(sizeY, blockH);
}

__device__ __host__ inline int MemoryMappedFrameBuffer::getBlockSize()const {
	return (blockW * blockH);
}
__device__ __host__ inline int MemoryMappedFrameBuffer::getBlockCount()const {
	return (widthBlocks() * heightBlocks());
}
__device__ __host__ inline bool MemoryMappedFrameBuffer::blockPixelLocation(int blockId, int pixelId, int *x, int *y)const {
	if (pixelId < 0 || pixelId >= getBlockSize()) return false;
	
	if (blockId < 0) return false;
	register int wBlocks = widthBlocks();
	if (wBlocks <= 0) return false;
	register int blockY = (blockId / wBlocks);
	if (blockY >= heightBlocks()) return false;
	register int blockX = (blockId - (blockY * wBlocks));
	
	register int shiftY = (pixelId / blockW);
	register int shiftX = (pixelId - (shiftY * blockW));
	register int posX = ((blockX * blockW) + shiftX);
	if (posX >= sizeX) return false;
	register int posY = ((blockY * blockH) + shiftY);
	if (posY >= sizeY) return false;

	(*x) = posX;
	(*y) = posY;
	return true;
}
__device__ __host__ inline bool MemoryMappedFrameBuffer::getBlockPixelColor(int blockId, int pixelId, Color *color)const {
	int x, y;
	if (!blockPixelLocation(blockId, pixelId, &x, &y)) return false;
	(*color) = getColor(x, y);
	return true;
}
__device__ __host__ inline bool MemoryMappedFrameBuffer::setBlockPixelColor(int blockId, int pixelId, const Color &color) {
	int x, y;
	if (!blockPixelLocation(blockId, pixelId, &x, &y)) return false;
	setColor(x, y, color);
	return true;
}
__device__ __host__ inline bool MemoryMappedFrameBuffer::blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount) {
	int x, y;
	if (!blockPixelLocation(blockId, pixelId, &x, &y)) return false;
	blendColor(x, y, color, amount);
	return true;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<>
__device__ __host__ inline void TypeTools<MemoryMappedFrameBuffer>::init(
	MemoryMappedFrameBuffer &m) {
	new(&m) MemoryMappedFrameBuffer();
}
template<>
__device__ __host__ inline void TypeTools<MemoryMappedFrameBuffer>::dispose(
	MemoryMappedFrameBuffer &m) {
	m.~MemoryMappedFrameBuffer();
}
template<>
__device__ __host__ inline void TypeTools<MemoryMappedFrameBuffer>::swap(
	MemoryMappedFrameBuffer &a, MemoryMappedFrameBuffer &b) {
	a.swapWith(b);
}
template<>
__device__ __host__ inline void TypeTools<MemoryMappedFrameBuffer>::transfer(
	MemoryMappedFrameBuffer &src, MemoryMappedFrameBuffer &dst) {
	dst.stealFrom(src);
}

template<>
inline bool TypeTools<MemoryMappedFrameBuffer>::prepareForCpyLoad(
	const MemoryMappedFrameBuffer *source,  MemoryMappedFrameBuffer *hosClone, 
	MemoryMappedFrameBuffer *, int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if ((source[i].flags & MemoryMappedFrameBuffer::CAN_NOT_USE_DEVICE) != 0) return false;
		else if (cudaHostGetDevicePointer(&hosClone[i].data, source[i].data, 0) != cudaSuccess) return false;
		hosClone[i].allocSize = source[i].allocSize;
		hosClone[i].sizeX = source[i].sizeX;
		hosClone[i].sizeY = source[i].sizeY;
		hosClone[i].blockW = source[i].blockW;
		hosClone[i].blockH = source[i].blockH;
		hosClone[i].flags = source[i].flags;
	}
	return true;
}
template<>
inline void TypeTools<MemoryMappedFrameBuffer>::undoCpyLoadPreparations(
	const MemoryMappedFrameBuffer *, MemoryMappedFrameBuffer *, MemoryMappedFrameBuffer *, int) { }
template<>
inline bool TypeTools<MemoryMappedFrameBuffer>::devArrayNeedsToBeDisposed() { return false; }
template<>
inline bool TypeTools<MemoryMappedFrameBuffer>::disposeDevArray(MemoryMappedFrameBuffer *, int) {
	return true;
}


