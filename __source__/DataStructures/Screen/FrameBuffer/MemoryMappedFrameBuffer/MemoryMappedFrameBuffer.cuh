#pragma once
#include"../../../Primitives/Pure/Color/Color.h"
#include"../../../GeneralPurpose/TypeTools/TypeTools.cuh"



class MemoryMappedFrameBuffer;
SPECIALISE_TYPE_TOOLS_FOR(MemoryMappedFrameBuffer);


class MemoryMappedFrameBuffer {
public:
	__device__ __host__ inline MemoryMappedFrameBuffer();
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
	inline bool updateBlocks(int startBlock, int endBlock, 
		int blockWidth, int blockHeight, const MemoryMappedFrameBuffer *deviceObject);
	
	__device__ __host__ inline void getSize(int &width, int &height)const;
	__device__ __host__ inline Color getColor(int x, int y)const;
	__device__ __host__ inline void setColor(int x, int y, const Color &color);
	__device__ __host__ inline void blendColor(int x, int y, const Color &color, float amount);
	__device__ __host__ inline Color* getData();
	__device__ __host__ inline const Color* getData()const;




private:
	enum Flags {
		CAN_NOT_USE_DEVICE = 1
	};
	int allocSize;
	int sizeX, sizeY;
	int flags;
	typedef Color* ColorPointer;
	ColorPointer data;

	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(MemoryMappedFrameBuffer);
};





#include"MemoryMappedFrameBuffer.impl.cuh"
