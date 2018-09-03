#include "Renderer.cuh"



Renderer::ThreadConfiguration::ThreadConfiguration() : ThreadConfiguration(ALL, ONE) { }
Renderer::ThreadConfiguration::ThreadConfiguration(int cpuThreads, int threadsPerDevice) {
	int deviceCount;
	if (cudaGetDeviceCount(&deviceCount) == cudaSuccess)
		for (int i = 0; i < deviceCount; i++)
			threadsPerGPU.push(0);
	flags = 0;
	configureCPU(cpuThreads);
	configureEveryGPU(threadsPerDevice);
}
Renderer::ThreadConfiguration Renderer::ThreadConfiguration::cpuOnly(int threads) {
	return ThreadConfiguration(threads, NONE);
}
Renderer::ThreadConfiguration Renderer::ThreadConfiguration::gpuOnly(int threads) {
	return ThreadConfiguration(NONE, threads);
}
void Renderer::ThreadConfiguration::configureCPU(int threads) {
	threadsOnCPU = ((threads >= 0) ? threads : std::thread::hardware_concurrency());
	if (threads == ALL_BUT_THREAD_PER_GPU) flags = 1;
	else if (threads == ALL_BUT_GPU_THREADS) flags = 2;
	else flags = 0;
	fixCpuThreadCount();
}
void Renderer::ThreadConfiguration::configureGPU(int GPU, int threads) {
	if (GPU < 0 || GPU >= threadsPerGPU.size()) return;
	threadsPerGPU[GPU] = ((threads >= 0) ? threads : (-threads));
	fixCpuThreadCount();
}
void Renderer::ThreadConfiguration::configureEveryGPU(int threads) {
	for (int i = 0; i < threadsPerGPU.size(); i++) configureGPU(i, threads);
	fixCpuThreadCount();
}

int Renderer::ThreadConfiguration::numActiveDevices()const {
	int num = 0;
	for (int i = 0; i < threadsPerGPU.size(); i++)
		if (threadsPerGPU[i] > 0) num++;
	return num;
}
int Renderer::ThreadConfiguration::numDevices()const {
	return threadsPerGPU.size();
}
int Renderer::ThreadConfiguration::numDeviceThreads(int deviceId)const {
	if (deviceId < 0 || deviceId >= threadsPerGPU.size()) return 0;
	else return threadsPerGPU[deviceId];
}
int Renderer::ThreadConfiguration::numDeviceThreads()const {
	int num = 0;
	for (int i = 0; i < threadsPerGPU.size(); i++)
		num += threadsPerGPU[i];
	return num;
}
int Renderer::ThreadConfiguration::numHostThreads()const {
	return threadsOnCPU;
}

void Renderer::ThreadConfiguration::fixCpuThreadCount() {
	if (flags == 0) return;
	else if (flags == 1) threadsOnCPU = (std::thread::hardware_concurrency() - numActiveDevices());
	else if (flags == 2) threadsOnCPU = (std::thread::hardware_concurrency() - numDeviceThreads());
}





#define DEVICE_CPU -1

Renderer::Renderer(const ThreadConfiguration &config) {
	configuration = config;

	cpuThreads = config.threadsOnCPU;
	gpuThreads = 0;
	for (int i = 0; i < config.threadsPerGPU.size(); i++)
		gpuThreads += config.threadsPerGPU[i];

	totalThreadCount = (cpuThreads + gpuThreads);
	if (totalThreadCount <= 0) {
		threads = NULL;
		gpuLocks = NULL;
	}
	else {
		threads = new Thread[totalThreadCount];
		gpuLocks = new Device[config.threadsPerGPU.size()];
	}


	if (!configured()) return;

	cpuLocks.firstThread = 0;
	cpuLocks.threadCount = cpuThreads;
	for (int i = 0; i < cpuThreads; i++) {
		threads[i].properties.info.device = DEVICE_CPU;
		threads[i].properties.info.numDeviceThreads = cpuThreads;
		threads[i].properties.info.deviceThreadId = i;
		threads[i].properties.info.globalThreadId = i;
		threads[i].properties.info.manageSharedData = (i == 0);
		threads[i].properties.device = (&cpuLocks);
		threads[i].properties.info.data = NULL;
		threads[i].properties.info.sharedData = NULL;
	}
	int pointer = cpuThreads;
	for (int i = 0; i < config.threadsPerGPU.size(); i++) {
		gpuLocks[i].firstThread = pointer;
		gpuLocks[i].threadCount = config.threadsPerGPU[i];
		for (int j = 0; j < config.threadsPerGPU[i]; j++) {
			threads[pointer].properties.info.device = i;
			threads[pointer].properties.info.numDeviceThreads = config.threadsPerGPU[i];
			threads[pointer].properties.info.deviceThreadId = j;
			threads[pointer].properties.info.globalThreadId = pointer;
			threads[pointer].properties.info.manageSharedData = (j == 0);
			threads[pointer].properties.device = (gpuLocks + i);
			threads[pointer].properties.info.data = NULL;
			threads[pointer].properties.info.sharedData = NULL;
			pointer++;
		}
	}
	command = ITERATE;
	for (int i = 0; i < totalThreadCount; i++)
		threads[i].thread = std::thread(thread, this, &threads[i].properties);
	destructorCalled = false;
	threadsStarted = false;
	resetIterations();
}
Renderer::~Renderer() {
	destructorCalled = true;
	killRenderThreads();
}
bool Renderer::configured()const {
	return (threads != NULL && gpuLocks != NULL);
}

int Renderer::iteration()const {
	return iterationId;
}
void Renderer::resetIterations() {
	iterationId = 0;
}
bool Renderer::iterate() {
	startRenderThreads();
	if (!threadsStarted) return false;
	iterationId++;
	if (!prepareIteration()) {
		iterationId--;
		return false;
	}
	for (int i = 0; i < totalThreadCount; i++)
		threads[i].properties.startLock.post();
	for (int i = 0; i < totalThreadCount; i++)
		threads[i].properties.endLock.wait();
	return completeIteration();
}

int Renderer::deviceThreadCount() { return gpuThreads; }

int Renderer::hostThreadCount() { return cpuThreads; }

Renderer::ThreadConfiguration &Renderer::threadConfiguration() { return configuration; }

const Renderer::ThreadConfiguration &Renderer::threadConfiguration()const { return configuration; }

void Renderer::interruptRender() { command |= INTERRUPT_RENDER; }

void Renderer::uninterruptRender() { command &= (~((uint8_t)INTERRUPT_RENDER)); }

bool Renderer::renderInterrupted()const { return ((command & INTERRUPT_RENDER) != 0); }

void Renderer::killRenderThreads() {
	startRenderThreads();
	if (configured()) {
		command |= QUIT;
		for (int i = 0; i < totalThreadCount; i++)
			threads[i].properties.startLock.post();
		for (int i = 0; i < totalThreadCount; i++)
			threads[i].thread.join();
	}
	if (threads != NULL) {
		delete[] threads;
		threads = NULL;
	}
	if (gpuLocks != NULL) {
		delete[] gpuLocks;
		gpuLocks = NULL;
	}
}


bool Renderer::Info::isGPU()const {
	return (device != DEVICE_CPU);
}



Renderer::Renderer(const Renderer &) { }
Renderer& Renderer::operator=(const Renderer &) { return (*this); }

void Renderer::startRenderThreads() {
	if ((!threadsStarted) && configured()) {
		threadsStarted = true;
		for (int i = 0; i < totalThreadCount; i++)
			threads[i].properties.startLock.post();
		for (int i = 0; i < totalThreadCount; i++)
			threads[i].properties.endLock.wait();
	}
}

void Renderer::threadCPU(ThreadAttributes *attributes) {
	bool canIterate = (!attributes->setupFailed);
	while (true) {
		attributes->startLock.wait();
		if ((command & QUIT) != 0) break;
		else if ((command & INTERRUPT_RENDER) != 0) {}
		else if (attributes->info.deviceThreadId >= configuration.numHostThreads()) {}
		else if (canIterate) iterateCPU(attributes->info);
		attributes->endLock.post();
	}
}
void Renderer::threadGPU(ThreadAttributes *attributes) {
	bool canIterate = ((!attributes->setupFailed) && (!attributes->device->setupFailed));
	while (true) {
		attributes->startLock.wait(); 
		if ((command & QUIT) != 0) break;
		else if ((command & INTERRUPT_RENDER) != 0) {}
		else if (attributes->info.deviceThreadId >= configuration.numDeviceThreads(attributes->info.device)) {}
		else if (canIterate) iterateGPU(attributes->info);
		attributes->endLock.post();
	}
}
void Renderer::thread(Renderer *renderer, ThreadAttributes *attributes) {
	attributes->startLock.wait();
	if (attributes->info.manageSharedData) {
		if (!renderer->destructorCalled)
			attributes->device->setupFailed = (!renderer->setupSharedData(attributes->info, attributes->info.sharedData));
		else attributes->device->setupFailed = true;
		if (!attributes->device->setupFailed)
			for (int i = 1; i < attributes->device->threadCount; i++)
				renderer->threads[i + attributes->device->firstThread].properties.info.sharedData = attributes->info.sharedData;
		for (int i = 0; i < attributes->info.numDeviceThreads; i++)
			attributes->device->setup.post();
	}
	attributes->device->setup.wait();
	if ((!attributes->device->setupFailed) && (!renderer->destructorCalled))
		attributes->setupFailed = (!renderer->setupData(attributes->info, attributes->info.data));
	else attributes->setupFailed = true;

	attributes->endLock.post();

	
	if (attributes->info.device == DEVICE_CPU) renderer->threadCPU(attributes);
	else renderer->threadGPU(attributes);
	
	
	if ((!attributes->device->setupFailed) && (!attributes->setupFailed) && (!renderer->destructorCalled))
		attributes->cleanFailed = (!renderer->clearData(attributes->info, attributes->info.data));
	else attributes->cleanFailed = false;
	attributes->device->clear.post();
	if (attributes->info.manageSharedData) {
		for (int i = 0; i < attributes->info.numDeviceThreads; i++)
			attributes->device->clear.wait();
		if (!attributes->device->setupFailed && (!renderer->destructorCalled))
			attributes->device->cleanFailed = (!renderer->clearSharedData(attributes->info, attributes->info.sharedData));
		else attributes->device->cleanFailed = false;
		if (!attributes->device->cleanFailed)
			for (int i = 1; i < attributes->device->threadCount; i++)
				renderer->threads[i + attributes->device->firstThread].properties.info.sharedData = attributes->info.sharedData;
	}
}
