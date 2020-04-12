#include "Checkerboard.cuh"
#include"../../DataStructures/Objects/Components/Lenses/DefaultPerspectiveLense/DefaultPerspectiveLense.cuh"
#include"../../DataStructures/Objects/Scene/Raycasters/ShadedOctree/ShadedOctree.cuh"
#include"../../DataStructures/Screen/FrameBuffer/MemoryMappedFrameBuffer/MemoryMappedFrameBuffer.cuh"
#include"../../DataStructures/Screen/FrameBuffer/FrameBuffer.cuh"
#include"../../Namespaces/Windows/Windows.h"
#include"../../DataStructures/Objects/Scene/Lights/SimpleDirectionalLight/SimpleDirectionalLight.cuh"
#include<time.h>


namespace {

#define HOR_SAMPLES 8
#define VER_SAMPLES 8

	__device__ __host__ inline void renderBlock(Color *color, int width, int height, int threadWidth, int threadHeight, int frameId, bool checkerboard, int blockX, int threadX) {
		int xStart = (blockX * threadWidth), yStart = (threadX * threadHeight);
		int xEnd = min((xStart + threadWidth), width), yEnd = min((yStart + threadHeight), height);
		float frameVal = (((float)frameId) / 256.0f);
		register int dj = (checkerboard ? 2 : 1);
		for (int i = yStart; i < yEnd; i++) {
			Color *line = (color + (width * i));
			for (int j = xStart + (checkerboard ? (i & 1) : 0); j < xEnd; j += dj) {
				Color c(0, 0, 0);
				for (float y = (float)i; y < (float)(i + 1); y += (1.0f / (float)VER_SAMPLES))
					for (float x = (float)j; x < (float)(j + 1); x += (1.0f / (float)HOR_SAMPLES))
						c += Color(
							cos(sin((y * x / frameVal) / 1024.0f + frameVal) + frameVal),
							sin(tan(sin((x / y * frameVal) / 1024.0f + frameVal)) * (frameVal + x + y) / 128.0f),
							cos(tan((x + y * frameVal) / 1024.0f * frameVal)) * frameVal);
				line[j] = (c / (HOR_SAMPLES * VER_SAMPLES));
			}
		}
	}
	__device__ __host__ inline void checkerboardFixBlock(Color *color, int width, int height, int threadWidth, int threadHeight, int blockX, int threadX) {
		int xStart = (blockX * threadWidth), yStart = (threadX * threadHeight);
		int xEnd = min((xStart + threadWidth), width), yEnd = min((yStart + threadHeight), height);
		for (int i = yStart; i < yEnd; i++) {
			Color *line = (color + (width * i));
			for (int j = xStart + (((i & 1) == 0) ? 1 : 0); j < xEnd; j += 2) {
				Color c(0, 0, 0);
				float count = 0;
				if (j > 0) { c += line[j - 1]; count++; }
				if (j < (width - 1)) { c += line[j + 1]; count++; }
				if (i > 0) { c += (line - 1)[j]; count++; }
				if (i < (height - 1)) { c += (line + 1)[j]; count++; }
				if (count > 0) c /= count;
				line[j] = c;
			}
		}
	}

	__global__ void render(Color *color, int width, int height, int threadWidth, int threadHeight, int frameId, bool checkerboard) {
		renderBlock(color, width, height, threadWidth, threadHeight, frameId, checkerboard, blockIdx.x, threadIdx.x);
	}
	__global__ void checkerboardFix(Color *color, int width, int height, int threadWidth, int threadHeight) {
		checkerboardFixBlock(color, width, height, threadWidth, threadHeight, blockIdx.x, threadIdx.x);
	}


#define CHUNK_COUNT(total, chunkSize) ((total + chunkSize - 1) / chunkSize)
	struct TestCheckerboardCPUThreadMainArgs {
		Color *color;
		int width;
		int height;
		int threadWidth;
		int threadHeight;
		volatile int *frameId;
		bool checkerboard;
		Semaphore *semaphore;
		Semaphore *release;
		std::mutex *blockLock;
		volatile int *numBlocks;
		volatile bool *killSwitch;
	};

	void testCheckerboardCPUThreadMain(TestCheckerboardCPUThreadMainArgs args) {
		int kernelWidth = CHUNK_COUNT(args.width, args.threadWidth);
		while (true) {
			args.semaphore->wait();
			if (!(*args.killSwitch)) while (true) {
				args.blockLock->lock();
				int blockId = ((*args.numBlocks) - 1);
				(*args.numBlocks)--;
				args.blockLock->unlock();
				if (blockId >= 0) {
					int blockY = (blockId / kernelWidth);
					int blockX = (blockId - (blockY * kernelWidth));
					renderBlock(args.color, args.width, args.height, args.threadWidth, args.threadHeight, *args.frameId, args.checkerboard, blockX, blockY);
				}
				else break;
			}
			args.release->post();
			if (*args.killSwitch) break;
		}
	}
	
	struct TestCheckerboardCPUThreadFixArgs {
		Color *color;
		int width;
		int height;
		int threadWidth;
		int threadHeight;
		volatile int *frameId;
		bool checkerboard;
		Semaphore *semaphore;
		Semaphore *release;
		std::mutex *blockLock;
		volatile int *numBlocks;
		volatile bool *killSwitch;
	};

	void testCheckerboardCPUThreadFix(TestCheckerboardCPUThreadFixArgs args) {
		int kernelWidth = CHUNK_COUNT(args.width, args.threadWidth);
		while (true) {
			args.semaphore->wait();
			if (!(*args.killSwitch)) while (true) {
				args.blockLock->lock();
				int blockId = ((*args.numBlocks) - 1);
				(*args.numBlocks)--;
				args.blockLock->unlock();
				if (blockId >= 0) {
					int blockY = (blockId / kernelWidth);
					int blockX = (blockId - (blockY * kernelWidth));
					if (args.checkerboard) checkerboardFixBlock(args.color, args.width, args.height, args.threadWidth, args.threadHeight, blockX, blockY);
				}
				else break;
			}
			args.release->post();
			if (*args.killSwitch) break;
		}
	}

	struct TestCheckerboardCPUArgs {
		int width;
		int height;
		int threadWidth;
		int threadHeight;
		bool checkerboard;
		std::mutex *ioLock;	
	};

	void testCheckerboardCPU(TestCheckerboardCPUArgs args) {
		int width = args.width;
		int height = args.height;
		int threadWidth = args.threadWidth;
		int threadHeight = args.threadHeight;
		bool checkerboard = args.checkerboard;
		std::mutex *ioLock = args.ioLock;
		Color *color = new Color[width * height];
		if (color == NULL) { std::cout << "ERROR(testCheckerboardCPU): Buffer allocation failed" << std::endl; return; }
		Windows::Window window((std::wstring(L"testCheckerboardCPU") + (checkerboard ? L" (checkerboard)" : L" (native)")).c_str());
		const int numThreads = std::thread::hardware_concurrency();
		std::thread *threads = new std::thread[numThreads];
		std::thread *fixThreads = new std::thread[numThreads];
		Semaphore *semaphores = new Semaphore[numThreads];
		Semaphore *fixSemaphores = new Semaphore[numThreads];
		Semaphore semaphore;
		std::mutex blockLock;
		volatile int numBlocks;
		int kernelWidth = CHUNK_COUNT(width, threadWidth);
		int kernelHeight = CHUNK_COUNT(height, threadHeight);
		volatile int frameId = 0;
		int lastCheckFrameId = frameId;
		time_t lastCheck = clock();
		volatile bool killSwitch = false;
		for (int i = 0; i < numThreads; i++) {
			threads[i] = std::thread(testCheckerboardCPUThreadMain, TestCheckerboardCPUThreadMainArgs {
				color, width, height, threadWidth, threadHeight, &frameId, checkerboard,
				semaphores + i, &semaphore, &blockLock, &numBlocks, &killSwitch });
			if (checkerboard) fixThreads[i] = std::thread(testCheckerboardCPUThreadFix, TestCheckerboardCPUThreadFixArgs {
				color, width, height, threadWidth, threadHeight, &frameId, checkerboard,
				fixSemaphores + i, &semaphore, &blockLock, &numBlocks, &killSwitch });
		}
		while (!window.dead()) {
			numBlocks = (kernelWidth * kernelHeight);
			for (int i = 0; i < numThreads; i++) semaphores[i].post();
			for (int i = 0; i < numThreads; i++) semaphore.wait();
			if (checkerboard) {
				numBlocks = (kernelWidth * kernelHeight);
				for (int i = 0; i < numThreads; i++) fixSemaphores[i].post();
				for (int i = 0; i < numThreads; i++) semaphore.wait();
			}
			window.updateFrameHost(color, width, height);
			frameId++;
			time_t now = clock();
			time_t deltaClock = (now - lastCheck);
			if (deltaClock >= CLOCKS_PER_SEC) {
				ioLock->lock();
				std::cout << "FPS: " << ((frameId - lastCheckFrameId) / (((float)deltaClock) / CLOCKS_PER_SEC)) << " (CPU)" << std::endl;
				ioLock->unlock();
				lastCheck = now;
				lastCheckFrameId = frameId;
			}
		}
		killSwitch = true;
		for (int i = 0; i < numThreads; i++) { semaphores[i].post(); fixSemaphores[i].post(); }
		for (int i = 0; i < numThreads; i++) { threads[i].join(); if (checkerboard) fixThreads[i].join(); }
		delete[] semaphores;
		delete[] fixSemaphores;
		delete[] threads;
		delete[] fixThreads;
		delete[] color;
	}

	
}

namespace CheckerboardTest {
	void test() {
		cudaStream_t stream[2];
		for (int i = 0; i < 2; i++)
			if (cudaStreamCreate(stream + i) != cudaSuccess) {
				std::cout << "ERROR(testCheckerboard): Cuda stream creation failed...." << std::endl;
				return;
			}
		int width = 1920, height = 1080;
		int threadWidth = 4, threadHeight = 4;
		Color *color[2] = { NULL, NULL };
		for (int i = 0; i < 2; i++)
			if (cudaMalloc(color + i, sizeof(Color) * width * height) != cudaSuccess) {
				std::cout << "ERROR(testCheckerboard): Color buffer allocation failed...." << std::endl;
				cudaStreamDestroy(stream[0]); cudaStreamDestroy(stream[1]);
				return;
			}
		Windows::Window checkerboard(L"testCheckerboard window (checkerboard)");
		Windows::Window native(L"testCheckerboard window (native)");
		std::mutex ioLock;
		std::thread cpuRenderer(testCheckerboardCPU, TestCheckerboardCPUArgs { width, height, threadWidth, threadHeight, true, &ioLock });
		int kernelWidth = CHUNK_COUNT(width, threadWidth);
		int kernelHeight = CHUNK_COUNT(height, threadHeight);
		int frameId = 0;
		int lastCheckFrameId = frameId;
		time_t lastCheck = clock();
		while ((!native.dead()) || (!checkerboard.dead())) {
			if (!native.dead()) render << <kernelWidth, kernelHeight, 0, stream[0] >> > (color[0], width, height, threadWidth, threadHeight, frameId, false);
			if (!checkerboard.dead()) {
				render << <kernelWidth, kernelHeight, 0, stream[1] >> > (color[1], width, height, threadWidth, threadHeight, frameId, true);
				checkerboardFix << <kernelWidth, kernelHeight, 0, stream[1] >> > (color[1], width, height, threadWidth, threadHeight);
			}
			bool success = (cudaStreamSynchronize(stream[0]) == cudaSuccess);
			if (!(cudaStreamSynchronize(stream[1]) == cudaSuccess)) success = false;
			if (!success) {
				std::cout << "ERROR(testCheckerboard): Stream synchronisation failed...." << std::endl;
				cudaStreamDestroy(stream[0]); cudaStreamDestroy(stream[1]);
				cudaFree(color[0]); cudaFree(color[1]);
				return;
			}
			frameId++;
			time_t now = clock();
			time_t deltaClock = (now - lastCheck);
			if (deltaClock >= CLOCKS_PER_SEC) {
				ioLock.lock();
				std::cout << "FPS: " << ((frameId - lastCheckFrameId) / (((float)deltaClock) / CLOCKS_PER_SEC)) << " (GPU)" << std::endl;
				ioLock.unlock();
				lastCheck = now;
				lastCheckFrameId = frameId;
			}
			native.updateFrameDevice(color[0], width, height);
			checkerboard.updateFrameDevice(color[1], width, height);
		}
		cudaStreamDestroy(stream[0]); cudaStreamDestroy(stream[1]);
		cudaFree(color[0]); cudaFree(color[1]);
		cudaFree(color);
		cpuRenderer.join();
	}
}
