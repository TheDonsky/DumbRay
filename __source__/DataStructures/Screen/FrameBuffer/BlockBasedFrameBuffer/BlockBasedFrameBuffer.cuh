#pragma once
#include"../../../Primitives/Pure/Color/Color.h"
#include"../../../GeneralPurpose/TypeTools/TypeTools.cuh"


class BlockBasedFrameBuffer;
SPECIALISE_TYPE_TOOLS_FOR(BlockBasedFrameBuffer);


class BlockBasedFrameBuffer {
public:
	__device__ __host__ inline BlockBasedFrameBuffer(int blockWidth = 16, int blockHeight = 16);
	__device__ __host__ inline ~BlockBasedFrameBuffer();

	inline BlockBasedFrameBuffer(const BlockBasedFrameBuffer &other);
	inline BlockBasedFrameBuffer& operator=(const BlockBasedFrameBuffer &other);
	inline bool copyFrom(const BlockBasedFrameBuffer &other);

	__device__ __host__ inline BlockBasedFrameBuffer(BlockBasedFrameBuffer &&other);
	__device__ __host__ inline BlockBasedFrameBuffer& operator=(BlockBasedFrameBuffer &&other);
	__device__ __host__ inline void stealFrom(const BlockBasedFrameBuffer &other);

	__device__ __host__ inline void swapWith(const BlockBasedFrameBuffer &other);


	__device__ __host__ inline void getSize(int *width, int *height)const;
	__device__ __host__ inline Color getColor(int x, int y)const;
	__device__ __host__ inline void setColor(int x, int y, const Color &color);
	__device__ __host__ inline void blendColor(int x, int y, const Color &color, float amount);

	__device__ __host__ inline int getBlockSize()const;
	__device__ __host__ inline int getBlockCount()const;
	__device__ __host__ inline bool blockPixelLocation(int blockId, int pixelId, int *x, int *y)const;
	__device__ __host__ inline bool getBlockPixelColor(int blockId, int pixelId, Color *color)const;
	__device__ __host__ inline bool setBlockPixelColor(int blockId, int pixelId, const Color &color);
	__device__ __host__ inline bool blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount);

	inline bool setResolution(int width, int height);
	inline static bool requiresBlockUpdate();
	inline bool updateDeviceInstance(BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream = NULL)const;
	inline bool updateBlocks(int startBlock, int endBlock, const BlockBasedFrameBuffer *deviceObject, cudaStream_t *stream = NULL);





private:
	int imageW, imageH;
	int blockW, blockH;
	int blockCount, blockSize;


	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(BlockBasedFrameBuffer);
};





#include "BlockBasedFrameBuffer.impl.cuh"
