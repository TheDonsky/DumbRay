#pragma once
#include"../../../Primitives/Pure/Color/Color.h"
#include"../../../GeneralPurpose/TypeTools/TypeTools.cuh"



class MemoryMappedFrameBuffer;
SPECIALISE_TYPE_TOOLS_FOR(MemoryMappedFrameBuffer);


class MemoryMappedFrameBuffer {
public:
	__device__ __host__ inline MemoryMappedFrameBuffer(int blockWidth = 16, int blockHeight=16);
	__device__ __host__ inline ~MemoryMappedFrameBuffer();
	__device__ __host__ inline void clear();

	inline MemoryMappedFrameBuffer(const MemoryMappedFrameBuffer &other);
	inline MemoryMappedFrameBuffer& operator=(const MemoryMappedFrameBuffer &other);
	inline bool copyFrom(const MemoryMappedFrameBuffer &other, bool accelerate = true);

	__device__ __host__ inline MemoryMappedFrameBuffer(MemoryMappedFrameBuffer &&other);
	__device__ __host__ inline MemoryMappedFrameBuffer& operator=(MemoryMappedFrameBuffer &&other);
	__device__ __host__ inline void stealFrom(MemoryMappedFrameBuffer &other);
	
	__device__ __host__ inline void swapWith(MemoryMappedFrameBuffer &other);

	inline bool setResolution(int width, int height);
	inline static bool requiresBlockUpdate();
	inline bool updateDeviceInstance(MemoryMappedFrameBuffer *deviceObject, cudaStream_t *stream)const;
	inline bool updateBlocks(int startBlock, int endBlock, const MemoryMappedFrameBuffer *deviceObject, cudaStream_t *stream);
	
	__device__ __host__ inline void getSize(int *width, int *height)const;
	__device__ __host__ inline Color getColor(int x, int y)const;
	__device__ __host__ inline void setColor(int x, int y, const Color &color);
	__device__ __host__ inline void blendColor(int x, int y, const Color &color, float amount);
	
	__device__ __host__ inline Color* getData();
	__device__ __host__ inline const Color* getData()const;

	__device__ __host__ inline static int blockCount(int dataSize, int blockSize);
	__device__ __host__ inline int widthBlocks()const;
	__device__ __host__ inline int heightBlocks()const;

	__device__ __host__ inline int getBlockSize()const;
	__device__ __host__ inline int getBlockCount()const;
	__device__ __host__ inline bool blockPixelLocation(int blockId, int pixelId, int *x, int *y)const;
	__device__ __host__ inline Color getBlockPixelColor(int blockId, int pixelId)const;
	__device__ __host__ inline void setBlockPixelColor(int blockId, int pixelId, const Color &color);
	__device__ __host__ inline void blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount);





private:
	enum Flags {
		CAN_NOT_USE_DEVICE = 1
	};
	int allocSize;
	int sizeX, sizeY;
	int blockW, blockH;
	int flags;
	typedef Color* ColorPointer;
	ColorPointer data;

	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(MemoryMappedFrameBuffer);
};





#include"MemoryMappedFrameBuffer.impl.cuh"
