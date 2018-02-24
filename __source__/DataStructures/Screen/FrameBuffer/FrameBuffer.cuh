#pragma once
#include"../../GeneralPurpose/Generic/Generic.cuh"
#include"../../Primitives/Pure/Color/Color.h"
#include<mutex>


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class FrameBufferFunctionPack {
public:
	__device__ __host__ inline FrameBufferFunctionPack();
	__device__ __host__ inline void clean();
	template<typename BufferType>
	__device__ __host__ inline void use();

	__device__ __host__ inline void getSize(const void *buffer, int *width, int *height)const;
	__device__ __host__ inline Color getColor(const void *buffer, int x, int y)const;
	__device__ __host__ inline void setColor(void *buffer, int x, int y, const Color &color)const;
	__device__ __host__ inline void blendColor(void *buffer, int x, int y, const Color &color, float amount)const;
	
	__device__ __host__ inline int getBlockSize(const void *buffer)const;
	__device__ __host__ inline int getBlockCount(const void *buffer)const;
	__device__ __host__ inline bool blockPixelLocation(const void *buffer, int blockId, int pixelId, int *x, int *y)const;
	__device__ __host__ inline bool getBlockPixelColor(const void *buffer, int blockId, int pixelId, Color *color)const;
	__device__ __host__ inline bool setBlockPixelColor(void *buffer, int blockId, int pixelId, const Color &color)const;
	__device__ __host__ inline bool blendBlockPixelColor(void *buffer, int blockId, int pixelId, const Color &color, float amount)const;


	inline bool setResolution(void *buffer, int width, int height)const;
	inline bool requiresBlockUpdate()const;
	inline bool updateDeviceInstance(const void *buffer, void *deviceObject)const;
	inline bool updateBlocks(void *buffer, int startBlock, int endBlock, const void *deviceObject)const;





private:
	void(*getSizeFn)(const void *buffer, int *width, int *height);
	Color(*getColorFn)(const void *buffer, int x, int y);
	void(*setColorFn)(void *buffer, int x, int y, const Color &color);
	void(*blendColorFn)(void *buffer, int x, int y, const Color &color, float amount);
	
	int(*getBlockSizeFn)(const void *buffer);
	int(*getBlockCountFn)(const void *buffer);
	bool(*blockPixelLocationFn)(const void *buffer, int blockId, int pixelId, int *x, int *y);
	bool(*getBlockPixelColorFn)(const void *buffer, int blockId, int pixelId, Color *color);
	bool(*setBlockPixelColorFn)(void *buffer, int blockId, int pixelId, const Color &color);
	bool(*blendBlockPixelColorFn)(void *buffer, int blockId, int pixelId, const Color &color, float amount);

	bool(*setResolutionFn)(void *buffer, int width, int height);
	bool(*requiresBlockUpdateFn)();
	bool(*updateDeviceInstanceFn)(const void *buffer, void *deviceObject);
	bool(*updateBlocksFn)(void *buffer, int startBlock, int endBlock, const void *deviceObject);

	template<typename BufferType>
	__device__ __host__ inline static void getSizeGeneric(const void *buffer, int *width, int *height);
	template<typename BufferType>
	__device__ __host__ inline static Color getColorGeneric(const void *buffer, int x, int y);
	template<typename BufferType>
	__device__ __host__ inline static void setColorGeneric(void *buffer, int x, int y, const Color &color);
	template<typename BufferType>
	__device__ __host__ inline static void blendColorGeneric(void *buffer, int x, int y, const Color &color, float amount);
	
	template<typename BufferType>
	__device__ __host__ inline static int getBlockSizeGeneric(const void *buffer);
	template<typename BufferType>
	__device__ __host__ inline static int getBlockCountGeneric(const void *buffer);
	template<typename BufferType>
	__device__ __host__ inline static bool blockPixelLocationGeneric(const void *buffer, int blockId, int pixelId, int *x, int *y);
	template<typename BufferType>
	__device__ __host__ inline static bool getBlockPixelColorGeneric(const void *buffer, int blockId, int pixelId, Color *color);
	template<typename BufferType>
	__device__ __host__ inline static bool setBlockPixelColorGeneric(void *buffer, int blockId, int pixelId, const Color &color);
	template<typename BufferType>
	__device__ __host__ inline static bool blendBlockPixelColorGeneric(void *buffer, int blockId, int pixelId, const Color &color, float amount);

	template<typename BufferType>
	inline static bool setResolutionGeneric(void *buffer, int width, int height);
	template<typename BufferType>
	inline static bool requiresBlockUpdateGeneric();
	template<typename BufferType>
	inline static bool updateDeviceInstanceGeneric(const void *buffer, void *deviceObject);
	template<typename BufferType>
	inline static bool updateBlocksGeneric(void *buffer, int startBlock, int endBlock, const void *deviceObject);
};






/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class FrameBuffer : public Generic<FrameBufferFunctionPack> {
public:
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
	inline bool requiresBlockUpdate()const;
	// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
	inline bool updateDeviceInstance(void *deviceObject)const;
	// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
	inline bool updateBlocks(int startBlock, int endBlock, const FrameBuffer *deviceObject);

	inline FrameBuffer *upload()const;
	inline static FrameBuffer* upload(const FrameBuffer *source, int count = 1);





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
	class BlockBank {
	public:
		inline BlockBank();
		inline void reset(const FrameBuffer &buffer);
		inline bool getBlocks(int count, int *start, int *end);



	private:
		int left;
		std::mutex lock;
	};
};



#include"FrameBuffer.impl.cuh"
