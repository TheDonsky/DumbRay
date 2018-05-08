#pragma once
#include"../../../Primitives/Pure/Color/Color.h"
#include"../../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include"../../../GeneralPurpose/TypeTools/TypeTools.cuh"
#include<mutex>
#include<unordered_map>


template<size_t blockW, size_t blockH> class BlockBasedFrameBuffer;
template<size_t blockW, size_t blockH>
class TypeTools<BlockBasedFrameBuffer<blockW, blockH> > {
public:
	typedef BlockBasedFrameBuffer<blockW, blockH> BufferType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(BufferType);
};


struct BlockBufferData {
private:

	Color *data;
	int imageW, imageH;
	int blockCount, allocCount;

	static std::mutex deviceReferenceLock;
	typedef std::unordered_map<const void*, BlockBufferData> DeviceReferenceMirrors;
	static DeviceReferenceMirrors deviceReferenceMirrors;

public:

	template<size_t blockW, size_t blockH> friend class BlockBasedFrameBuffer;
};

template<size_t blockW = 16, size_t blockH = blockW>
class BlockBasedFrameBuffer {
public:
	__device__ __host__ inline BlockBasedFrameBuffer();
	__device__ __host__ inline ~BlockBasedFrameBuffer();
	__device__ __host__ inline void clear();

	inline BlockBasedFrameBuffer(const BlockBasedFrameBuffer &other);
	inline BlockBasedFrameBuffer& operator=(const BlockBasedFrameBuffer &other);
	inline void copyFrom(const BlockBasedFrameBuffer &other);

	__device__ __host__ inline BlockBasedFrameBuffer(BlockBasedFrameBuffer &&other);
	__device__ __host__ inline BlockBasedFrameBuffer& operator=(BlockBasedFrameBuffer &&other);
	__device__ __host__ inline void stealFrom(BlockBasedFrameBuffer &other);

	__device__ __host__ inline void swapWith(BlockBasedFrameBuffer &other);


	__device__ __host__ inline void getSize(int *width, int *height)const;
	__device__ __host__ inline Color getColor(int x, int y)const;
	__device__ __host__ inline void setColor(int x, int y, const Color &color);
	__device__ __host__ inline void blendColor(int x, int y, const Color &color, float amount);

	__device__ __host__ inline int getBlockSize()const;
	__device__ __host__ inline int getBlockCount()const;
	__device__ __host__ inline bool pixelBlockLocation(int x, int y, int *blockId, int *pixelId)const;
	__device__ __host__ inline bool blockPixelLocation(int blockId, int pixelId, int *x, int *y)const;
	__device__ __host__ inline Color getBlockPixelColor(int blockId, int pixelId)const;
	__device__ __host__ inline void setBlockPixelColor(int blockId, int pixelId, const Color &color);
	__device__ __host__ inline void blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount);

	inline bool setResolution(int width, int height);
	inline bool updateDeviceBlocks(BlockBasedFrameBuffer *deviceBuffer, int startBlock, int endBlock, cudaStream_t *stream)const;
	inline bool updateHostBlocks(const BlockBasedFrameBuffer *deviceBuffer, int startBlock, int endBlock, cudaStream_t *stream);





private:
	BlockBufferData buffer;

	inline Color*& data() { return buffer.data; }
	inline const Color* data()const { return buffer.data; }

	inline int& allocCount() { return buffer.allocCount; }
	inline const int& allocCount()const { return buffer.allocCount; }

	inline static std::mutex &deviceReferenceLock() { return BlockBufferData::deviceReferenceLock; }
	typedef BlockBufferData::DeviceReferenceMirrors DeviceReferenceMirrors;
	inline static DeviceReferenceMirrors &deviceReferenceMirrors() { return BlockBufferData::deviceReferenceMirrors; }


	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(BlockBasedFrameBuffer);
};


typedef BlockBasedFrameBuffer<16, 16> BlockBuffer;





#include "BlockBasedFrameBuffer.impl.cuh"

