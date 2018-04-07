#pragma once
#include"../../../Primitives/Pure/Color/Color.h"
#include"../../../Primitives/Compound/Pair/Pair.cuh"
#include"../../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include"../../../GeneralPurpose/TypeTools/TypeTools.cuh"
#include<mutex>
#include<unordered_map>


class BlockBasedFrameBuffer;
SPECIALISE_TYPE_TOOLS_FOR(BlockBasedFrameBuffer);


class BlockBasedFrameBuffer {
public:
	__device__ __host__ inline BlockBasedFrameBuffer(int blockWidth = 16, int blockHeight = 16);
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
	__device__ __host__ inline bool getBlockPixelColor(int blockId, int pixelId, Color *color)const;
	__device__ __host__ inline bool setBlockPixelColor(int blockId, int pixelId, const Color &color);
	__device__ __host__ inline bool blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount);

	inline bool setResolution(int width, int height);
	inline static bool requiresBlockUpdate();
	inline bool updateDeviceInstance(BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream = NULL)const;
	inline bool updateBlocks(int startBlock, int endBlock, const BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream = NULL);





private:
	struct BufferData {
		int blockW, blockH, blockSize;

		int imageW, imageH;
		int blockCount, allocCount;
		Color *data;
	};
	BufferData buffer;

	static std::mutex deviceReferenceLock;
	typedef std::unordered_map<const BlockBasedFrameBuffer*, BufferData> DeviceReferenceMirrors;
	static DeviceReferenceMirrors deviceReferenceMirrors;


	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(BlockBasedFrameBuffer);
};





#include "BlockBasedFrameBuffer.impl.cuh"
