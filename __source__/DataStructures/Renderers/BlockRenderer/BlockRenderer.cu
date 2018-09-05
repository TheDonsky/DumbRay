#include "BlockRenderer.cuh"


BlockRenderer::BlockConfiguration::BlockConfiguration(int blockCutPerCpuThread, int blockCutPerGpuSM, bool forceDeviceInstanceUpdate) {
	blocksPerCpuThread = blockCutPerCpuThread;
	blocksPerGpuSM = blockCutPerGpuSM;
	forceUpdateDeviceInstance = forceDeviceInstanceUpdate;
}

int BlockRenderer::BlockConfiguration::blockCutPerCpuThread()const { return blocksPerCpuThread; }
int BlockRenderer::BlockConfiguration::blockCutPerGpuSM()const { return blocksPerGpuSM; }
bool BlockRenderer::BlockConfiguration::forceDeviceInstanceUpdate()const { return forceUpdateDeviceInstance; }


BlockRenderer::BlockRenderer(
	const ThreadConfiguration &configuration, 
	const BlockConfiguration &blockSettings, 
	FrameBufferManager *buffer) : BufferedRenderer(configuration, buffer) {

	blockConfiguration = blockSettings;
	threadData.flush(deviceThreadCount() + hostThreadCount());
	hostBlockSynchNeeded = (
		(threadConfiguration().numActiveDevices() > 1) 
		|| (threadConfiguration().numHostThreads() > 0)
		|| blockConfiguration.forceDeviceInstanceUpdate());
}
BlockRenderer::~BlockRenderer() { killRenderThreads(); }

const BlockRenderer::BlockConfiguration &BlockRenderer::blockRendererConfiguration() const { return blockConfiguration; }
bool BlockRenderer::automaticallySynchesHostBlocks()const { return hostBlockSynchNeeded; }


bool BlockRenderer::setupSharedData(const Info &, void *&) { 
	/*
	size_t stackSize;
	if (cudaDeviceGetLimit(&stackSize, cudaLimitStackSize) != cudaSuccess) return false;
	const int neededStackSize = 8192;
	if (stackSize < neededStackSize) if (cudaDeviceSetLimit(cudaLimitStackSize, neededStackSize) != cudaSuccess) return false;
	//*/
	return true; 
}
bool BlockRenderer::setupData(const Info &info, void *&) {
	// __TODO__: (maybe) record the errors somehow...
	if (info.isGPU()) {
		FrameBuffer::DeviceBlockManager *manager = new FrameBuffer::DeviceBlockManager(
			info.device, (hostBlockSynchNeeded ?
				(FrameBuffer::DeviceBlockManager::Settings)FrameBuffer::DeviceBlockManager::CUDA_RENDER_STREAM_AUTO_SYNCH_ON_GET :
				(FrameBuffer::DeviceBlockManager::Settings)FrameBuffer::DeviceBlockManager::CUDA_MANUALLY_SYNCH_HOST_BLOCKS),
			blockConfiguration.blockCutPerGpuSM());
		if (manager == NULL) return false;	// ALLOCATION FAILURE...
		else if (manager->errors() != 0) { delete manager; return false; }	// INTERNAL ERRORS...
		threadData[info.globalThreadId].blockManager = manager;
	}
	return true;
}
bool BlockRenderer::prepareIteration() {
	// __TODO__: (maybe) record the errors somehow...
	FrameBufferManager *manager = getFrameBuffer();
	if (manager == NULL) return false;
	FrameBuffer *cpuHandle = manager->cpuHandle();
	if (cpuHandle == NULL) return false;
	else if (cpuHandle->object() == NULL) return false;
	blockBank.reset(*cpuHandle);
	return true;
}
void BlockRenderer::iterateCPU(const Info &info) {
	// __TODO__: (maybe) record the errors somehow...
	FrameBuffer *buffer = getFrameBuffer()->cpuHandle();
	if (buffer == NULL) return;
	int start, end;
	while ((!renderInterrupted()) && blockBank.getBlocks(4, &start, &end))
		if (!renderBlocksCPU(info, buffer, start, end)) return;
}
void BlockRenderer::iterateGPU(const Info &info) {
	// __TODO__: (maybe) record the errors somehow...
	FrameBuffer *host = getFrameBuffer()->cpuHandle();
	if (host == NULL) return;
	FrameBuffer *device = getFrameBuffer()->gpuHandle(info.device);
	if (device == NULL) return;

	FrameBuffer::DeviceBlockManager *blockManager = threadData[info.globalThreadId].blockManager;
	if (blockManager == NULL) return; // NORMALLY, THIS SHOULD NOT HAPPEN AT ALL...
	if (!blockManager->setBuffers(host, device, &blockBank)) return;

	bool synchNeeded = ((iteration() > 1) && hostBlockSynchNeeded);
	int start = 0, end = 0;
	cudaStream_t &renderStream = blockManager->getRenderStream();
	while ((!renderInterrupted()) && blockManager->getBlocks(start, end, synchNeeded))
		if (!renderBlocksGPU(info, host, device, start, end, renderStream)) {
			if (cudaGetLastError() != cudaSuccess) threadConfiguration().configureGPU(info.device, 0);
			return;
		}

	// __TODO__: record errors if (blockManager->errors() != 0) 
	if (hostBlockSynchNeeded) { blockManager->synchBlockSynchStream(); /* THIS MAY FAIL AS WELL.. */ }
	else blockManager->synchRenderStream(); /* THIS MAY FAIL AS WELL.. */
}
bool BlockRenderer::completeIteration() {
	// __TODO__: (maybe) record the errors somehow...
	// __TODO__: return false if any error was detected...
	return true;
}
bool BlockRenderer::clearData(const Info &info, void *&) {
	// __TODO__: (maybe) record the errors somehow...
	if (info.isGPU()) {
		if (threadData[info.globalThreadId].blockManager == NULL) return false;
		delete threadData[info.globalThreadId].blockManager;
		threadData[info.globalThreadId].blockManager = NULL;
	}
	return true;
}
bool BlockRenderer::clearSharedData(const Info &, void *&) { return true; }
