#include "BlockBasedFrameBuffer.cuh"


__device__ __host__ inline BlockBasedFrameBuffer::BlockBasedFrameBuffer(int blockWidth, int blockHeight) {
	// __TODO__...
	imageW = imageH = 0;
	blockW = blockWidth;
	blockH = blockHeight;
	blockSize = (blockW * blockH);
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
	(*width) = imageW;
	(*height) = imageH;
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
	return blockSize;
}
__device__ __host__ inline int BlockBasedFrameBuffer::getBlockCount()const {
	// __TODO__...
	return blockCount;
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
	imageW = width;
	imageH = height;
	blockCount = (((imageW + blockW - 1) / blockW) * ((imageH + blockH - 1) / blockH));
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
	int i = 0;
	for (i = 0; i < count; i++) {
		// __TODO__....
	}
	return true;
}
template<>
inline void TypeTools<BlockBasedFrameBuffer>::undoCpyLoadPreparations(
	const BlockBasedFrameBuffer *, BlockBasedFrameBuffer *, BlockBasedFrameBuffer *, int) { 
	// __TODO__...
}
template<>
inline bool TypeTools<BlockBasedFrameBuffer>::devArrayNeedsToBeDisposed() { return true; }
template<>
inline bool TypeTools<BlockBasedFrameBuffer>::disposeDevArray(BlockBasedFrameBuffer *, int) {
	// __TODO__...
	return true;
}


