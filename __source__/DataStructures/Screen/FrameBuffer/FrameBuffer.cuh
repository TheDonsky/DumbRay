#pragma once
#include"../../GeneralPurpose/Generic/Generic.cuh"
#include"../../Primitives/Pure/Color/Color.h"


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class FrameBufferFunctionPack {
public:
	__device__ __host__ inline FrameBufferFunctionPack();
	__device__ __host__ inline void clean();
	template<typename BufferType>
	__device__ __host__ inline void use();

	__device__ __host__ inline void getSize(const void *buffer, int &width, int &height)const;
	__device__ __host__ inline Color getColor(const void *buffer, int x, int y)const;
	__device__ __host__ inline void setColor(void *buffer, int x, int y, const Color &color)const;
	__device__ __host__ inline void blendColor(void *buffer, int x, int y, const Color &color, float amount)const;
	__device__ __host__ inline Color* getData(void *buffer)const;
	__device__ __host__ inline const Color* getData(const void *buffer)const;

	inline bool setResolution(void *buffer, int width, int height);
	inline bool requiresBlockUpdate();
	inline bool updateBlocks(void *buffer, int startBlock, int endBlock,
		int blockWidth, int blockHeight, const void *deviceObject);





private:
	void(*getSizeFunction)(const void *buffer, int &width, int &height);
	Color(*getColorFunction)(const void *buffer, int x, int y);
	void(*setColorFunction)(void *buffer, int x, int y, const Color &color);
	void(*blendColorFunction)(void *buffer, int x, int y, const Color &color, float amount);
	Color*(*getDataFunction)(void *buffer);
	const Color*(*getDataFunctionConst)(const void *buffer);
	bool(*setResolutionFunction)(void *buffer, int width, int height);
	bool(*requiresBlockUpdateFunction)();
	bool(*updateBlocksFunction)(void *buffer, int startBlock, int endBlock,
		int blockWidth, int blockHeight, const void *deviceObject);

	template<typename BufferType>
	__device__ __host__ inline static void getSizeGeneric(const void *buffer, int &width, int &height);
	template<typename BufferType>
	__device__ __host__ inline static Color getColorGeneric(const void *buffer, int x, int y);
	template<typename BufferType>
	__device__ __host__ inline static void setColorGeneric(void *buffer, int x, int y, const Color &color);
	template<typename BufferType>
	__device__ __host__ inline static void blendColorGeneric(void *buffer, int x, int y, const Color &color, float amount);
	template<typename BufferType>
	__device__ __host__ inline static Color* getDataGeneric(void *buffer);
	template<typename BufferType>
	__device__ __host__ inline static const Color* getDataGenericConst(const void *buffer);
	template<typename BufferType>
	inline static bool setResolutionGeneric(void *buffer, int width, int height);
	template<typename BufferType>
	inline static bool requiresBlockUpdateGeneric();
	template<typename BufferType>
	inline static bool updateBlocksGeneric(void *buffer, int startBlock, int endBlock,
		int blockWidth, int blockHeight, const void *deviceObject);
};






/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class FrameBuffer : public Generic<FrameBufferFunctionPack> {
public:
	__device__ __host__ inline void getSize(int &width, int &height)const;
	__device__ __host__ inline Color getColor(int x, int y)const;
	__device__ __host__ inline void setColor(int x, int y, const Color &color);
	__device__ __host__ inline void blendColor(int x, int y, const Color &color, float amount);
	__device__ __host__ inline Color* getData();
	__device__ __host__ inline const Color* getData()const;

	inline bool setResolution(int width, int height);
	// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
	inline bool updateBlocks(int startBlock, int endBlock,
		int blockWidth, int blockHeight, const FrameBuffer *deviceObject);
	// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
	inline bool updateBlocks(int startBlock, int endBlock,
		int blockWidth, int blockHeight, const void *deviceObject);

	inline FrameBuffer *upload()const;
	inline static FrameBuffer* upload(const FrameBuffer *source, int count = 1);
};



#include"FrameBuffer.impl.cuh"
