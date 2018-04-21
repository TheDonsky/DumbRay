#pragma once
#include "../BufferedRenderer/BufferedRenderer.cuh"


class BlockRenderer : public BufferedRenderer {
public:
	class BlockConfiguration {
	public:
		BlockConfiguration(
			int blockCutPerCpuThread = 1, int blockCutPerGpuSM = 16,
			bool forceDeviceInstanceUpdate = false);

		int blockCutPerCpuThread()const;
		int blockCutPerGpuSM()const;
		bool forceDeviceInstanceUpdate()const;

	private:
		int blocksPerCpuThread, blocksPerGpuSM;
		bool forceUpdateDeviceInstance;
	};

public:
	BlockRenderer(
		const ThreadConfiguration &configuration = ThreadConfiguration(ThreadConfiguration::ALL, 2),
		const BlockConfiguration &blockSettings = BlockConfiguration(), 
		FrameBufferManager *buffer = NULL);
	virtual ~BlockRenderer();

	const BlockConfiguration &blockRendererConfiguration()const;
	bool automaticallySynchesHostBlocks()const;


protected:
	// USING THESE INSTEAD OF setupData MIGHT BE ADVICED:
	template<typename Type, typename... DefaultArgs>
	// Use this when you're 100% sure, no other type will be used in the system:
	inline Type *getThreadData(const Info &info, DefaultArgs... defaultArgs);
	template<typename Type, typename... DefaultArgs>
	// If there's a chance, the thread data type may change, use this one and this one only: 
	inline Type *getThreadDataSafe(const Info &info, DefaultArgs... defaultArgs);

	// OVERRIDE THESE TWO AND IMPLEMENT YOUR RENDERING LOGIC IN THEM:
	virtual bool renderBlocksCPU(const Info &info, FrameBuffer *buffer, int startBlock, int endBlock) = 0;
	virtual bool renderBlocksGPU(const Info &info, FrameBuffer *host, FrameBuffer *device, int startBlock, int endBlock, cudaStream_t &renderStream) = 0;





protected:
	// EVERY SINGLE OF THESE HAV TO BE CALLED ANYWAY, IF OVERRIDED...
	virtual bool setupSharedData(const Info &info, void *& sharedData);
	virtual bool setupData(const Info &info, void *& data);
	virtual bool prepareIteration();
	virtual void iterateCPU(const Info &info);
	virtual void iterateGPU(const Info &info);
	virtual bool completeIteration();
	virtual bool clearData(const Info &info, void *& data);
	virtual bool clearSharedData(const Info &info, void *& sharedData);




private:
	struct ThreadData {
		FrameBuffer::DeviceBlockManager *blockManager;
		
		typedef void(*DeleteFunction)(void *&object);
		void *object;
		DeleteFunction deleteFunction;
		template<typename Type>
		inline static void deleteObject(void *&object) {
			Type *type = ((Type*)object);
			if (type != NULL) {
				delete type;
				object = NULL;
			}
		}

		inline ThreadData() { blockManager = NULL; object = NULL; deleteFunction = NULL; }
		inline ~ThreadData() { 
			if (blockManager != NULL) delete blockManager; 
			if (deleteFunction != NULL) deleteFunction(object);
		}

		template<typename Type, typename... DefaultArgs>
		inline Type *getObject(DefaultArgs... defaultArgs) {
			if (object == NULL) {
				object = ((void*)new Type(defaultArgs...));
				deleteFunction = deleteObject<Type>();
			}
			return object;
		}

		template<typename Type, typename... DefaultArgs>
		inline Type *getObjectSafe(DefaultArgs... defaultArgs) {
			DeleteFunction newDeleteFunction = deleteObject<Type>();
			if (newDeleteFunction != deleteFunction) {
				if (deleteFunction != NULL) deleteFunction(object);
				object = ((void*)new Type(defaultArgs...));
				deleteFunction = newDeleteFunction;
			}
			return ((Type*)object);
		}
	};

	Stacktor<ThreadData> threadData;

	FrameBuffer::BlockBank blockBank;

	bool hostBlockSynchNeeded;

	BlockConfiguration blockConfiguration;
};





template<typename Type, typename... DefaultArgs>
inline Type *BlockRenderer::getThreadData(const Info &info, DefaultArgs... defaultArgs) {
	return threadData[info.globalThreadId].getObject(defaultArgs...);
}
template<typename Type, typename... DefaultArgs>
inline Type *BlockRenderer::getThreadDataSafe(const Info &info, DefaultArgs... defaultArgs) {
	return threadData[info.globalThreadId].getObjectSafe(defaultArgs...);
}
