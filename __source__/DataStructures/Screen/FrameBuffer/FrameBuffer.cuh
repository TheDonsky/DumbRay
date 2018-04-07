#pragma once
#include"../../GeneralPurpose/Generic/Generic.cuh"
#include"../../Primitives/Pure/Color/Color.h"
#include"../../Primitives/Compound/Pair/Pair.cuh"
#include"../../../Namespaces/Device/Device.cuh"
#include"../../GeneralPurpose/Semaphore/Semaphore.h"
#include<mutex>
#include<atomic>



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
	__device__ __host__ inline Color getBlockPixelColor(const void *buffer, int blockId, int pixelId)const;
	__device__ __host__ inline void setBlockPixelColor(void *buffer, int blockId, int pixelId, const Color &color)const;
	__device__ __host__ inline void blendBlockPixelColor(void *buffer, int blockId, int pixelId, const Color &color, float amount)const;


	inline bool setResolution(void *buffer, int width, int height)const;
	inline bool requiresBlockUpdate()const;
	inline bool updateDeviceInstance(const void *buffer, void *deviceObject, cudaStream_t *stream)const;
	inline bool updateBlocks(void *buffer, int startBlock, int endBlock, const void *deviceObject, cudaStream_t *stream)const;





private:
	void(*getSizeFn)(const void *buffer, int *width, int *height);
	Color(*getColorFn)(const void *buffer, int x, int y);
	void(*setColorFn)(void *buffer, int x, int y, const Color &color);
	void(*blendColorFn)(void *buffer, int x, int y, const Color &color, float amount);
	
	int(*getBlockSizeFn)(const void *buffer);
	int(*getBlockCountFn)(const void *buffer);
	bool(*blockPixelLocationFn)(const void *buffer, int blockId, int pixelId, int *x, int *y);
	Color(*getBlockPixelColorFn)(const void *buffer, int blockId, int pixelId);
	void(*setBlockPixelColorFn)(void *buffer, int blockId, int pixelId, const Color &color);
	void(*blendBlockPixelColorFn)(void *buffer, int blockId, int pixelId, const Color &color, float amount);

	bool(*setResolutionFn)(void *buffer, int width, int height);
	bool(*requiresBlockUpdateFn)();
	bool(*updateDeviceInstanceFn)(const void *buffer, void *deviceObject, cudaStream_t *stream);
	bool(*updateBlocksFn)(void *buffer, int startBlock, int endBlock, const void *deviceObject, cudaStream_t *stream);

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
	__device__ __host__ inline static Color getBlockPixelColorGeneric(const void *buffer, int blockId, int pixelId);
	template<typename BufferType>
	__device__ __host__ inline static void setBlockPixelColorGeneric(void *buffer, int blockId, int pixelId, const Color &color);
	template<typename BufferType>
	__device__ __host__ inline static void blendBlockPixelColorGeneric(void *buffer, int blockId, int pixelId, const Color &color, float amount);

	template<typename BufferType>
	inline static bool setResolutionGeneric(void *buffer, int width, int height);
	template<typename BufferType>
	inline static bool requiresBlockUpdateGeneric();
	template<typename BufferType>
	inline static bool updateDeviceInstanceGeneric(const void *buffer, void *deviceObject, cudaStream_t *stream);
	template<typename BufferType>
	inline static bool updateBlocksGeneric(void *buffer, int startBlock, int endBlock, const void *deviceObject, cudaStream_t *stream);
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
	__device__ __host__ inline Color getBlockPixelColor(int blockId, int pixelId)const;
	__device__ __host__ inline void setBlockPixelColor(int blockId, int pixelId, const Color &color);
	__device__ __host__ inline void blendBlockPixelColor(int blockId, int pixelId, const Color &color, float amount);

	inline bool setResolution(int width, int height);
	inline static void *getDeviceObject(const FrameBuffer *deviceBuffer, cudaStream_t *stream = NULL);
	inline bool requiresBlockUpdate()const;
	// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
	inline bool updateDeviceInstance(FrameBuffer *deviceObject, cudaStream_t *stream = NULL)const;
	// Note: deviceObject is meant to be of the same type, this FrameBuffer is actually using.
	inline bool updateBlocks(int startBlock, int endBlock, const FrameBuffer *deviceObject, cudaStream_t *stream = NULL);

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
		std::atomic<int> left;
	};





	class DeviceBlockManager {
	public:
		struct DeviceFrameSyncher {
		public:
			friend class DeviceBlockManager;

		private:
			Semaphore semaphore;
		};

		typedef uint16_t Settings, Errors, Status;

		enum SettingFlags {
			CUDA_BLOCK_SYNCH_STREAM_AUTO_SYNCH_ON_GET = 1,
			CUDA_RENDER_STREAM_AUTO_SYNCH_ON_GET = 2
		};

		enum ErrorFlags {
			CUDA_SET_DEVICE_FAILED = 1,
			CUDA_GET_SM_COUNT_FAILED = 2,
			CUDA_HOST_ALLOC_FAILED = 4,
			CUDA_SYNCH_STREAM_CREATE_FAILED = 8,
			CUDA_RENDER_STREAM_CREATE_FAILED = 16,
			CUDA_DEVICE_BUFFER_OBJECT_OBTAIN_FAILED = 32,
			CUDA_HOST_BLOCK_UPDATE_FAILED = 64,
			CUDA_SYNCH_STREAM_CYNCH_FAILED = 128,
			CUDA_RENDER_STREAM_SYNCH_FAILED = 256,
			CUDA_UPDATE_DEVICE_INSTANCE_FAILED = 512
		};

		enum StatusFlags {
			CUDA_SET_DEVICE_OK = 1,
			CUDA_GET_SM_COUNT_OK = 2,
			CUDA_HOST_ALLOC_OK = 4,
			CUDA_SYNCH_STREAM_CREATE_OK = 8,
			CUDA_RENDER_STREAM_CREATE_OK = 16,
			CUDA_DEVICE_BUFFER_PRESENT = 32
		};

		inline DeviceBlockManager(
			int deviceId, DeviceFrameSyncher *syncher,
			Settings settings, int blocksPerSM,
			size_t mmapedMemorySize = 64);
		inline ~DeviceBlockManager();

		inline Settings settings()const;
		inline Errors errors()const;
		inline Status status()const;

		inline bool setBuffers(FrameBuffer *host, FrameBuffer *device, BlockBank *bank);

		inline bool getBlocks(int &start, int &end);

		inline cudaStream_t &getBlockSynchStream();
		inline bool synchBlockSynchStream();
		inline cudaStream_t &getRenderStream();
		inline bool synchRenderStream();

		inline bool sunchDeviceInstance(bool isMasterThread, int otherThreadCount);





	private:
		Settings settingFlags;
		Errors errorFlags;
		Status statusFlags;
		
		int devId;
		int numSM;
		int batchBlocks;
		size_t mmapedByteCount;
		void *mmapedBytes;

		DeviceFrameSyncher *synch;
		cudaStream_t synchStream;
		cudaStream_t renderStream;

		void *deviceBufferObject;
		FrameBuffer *hostBuffer, *deviceBuffer;
		
		BlockBank *blockBank;
		int lastStartBlock, lastEndBlock;
	};
};



#include"FrameBuffer.impl.cuh"
