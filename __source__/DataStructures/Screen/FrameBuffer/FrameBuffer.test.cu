#include"FrameBuffer.test.cuh"
#include"../../Renderers/Renderer/Renderer.cuh"
#include"../../../Namespaces/Windows/Windows.h"
#include"../../../Namespaces/Device/Device.cuh"
#include"../../Primitives/Compound/Pair/Pair.cuh"
#include<mutex>
#include<sstream>


namespace FrameBufferTest {
	namespace {
		typedef unsigned long long Count;

		void frameBufferWindowThread(
			const FrameBufferManager **buffer, Windows::Window *window, 
			std::mutex *bufferLock, std::condition_variable *bufferLockCond,
			volatile Count *displayedFrameCount, volatile bool *shouldStop, volatile bool *stopped) {
			while (true) {
				std::unique_lock<std::mutex> uniqueLock(*bufferLock);
				bufferLockCond->wait(uniqueLock);
				if (*shouldStop) {
					(*stopped) = true;
					break;
				}
				window->updateFromHost(*(*buffer)->cpuHandle());
				(*displayedFrameCount)++;
			}
		}

		static void trySwapBuffers(
			FrameBufferManager **front, FrameBufferManager **back,
			std::mutex *bufferLock, std::condition_variable *swapCond) {
			if (bufferLock->try_lock()) {
				FrameBufferManager *tmp = (*front);
				(*front) = (*back);
				(*back) = tmp;
				swapCond->notify_all();
				bufferLock->unlock();
			}
		}

		__device__ __host__ inline void colorPixel(int blockId, int blockSize, int pixelId, FrameBuffer *buffer, int iteration) {
#ifdef __CUDA_ARCH__
			Color color(0, 1, 0, 1);
#else
			Color color(0, 0, 1, 1);
#endif
			color *= (((float)pixelId) / ((float)blockSize));
			if (iteration > 1) buffer->blendBlockPixelColor(blockId, pixelId, color, (1.0f / ((float)iteration)));
			else buffer->setBlockPixelColor(blockId, pixelId, color);
		}

		__global__ static void renderBlocks(int startBlock, FrameBuffer *buffer, int iteration) {
			colorPixel(startBlock + blockIdx.x, blockDim.x, threadIdx.x, buffer, iteration);
		}

		template<typename First>
		static void printToStream(std::ostream &stream, First first) {
			stream << first;
		}
		template<typename First, typename... Rest>
		static void printToStream(std::ostream &stream, First first, Rest... rest) {
			printToStream(stream << first, rest...);
		}

		template<typename... Types>
		static void print(Types... types) {
			std::stringstream stream;
			printToStream(stream, types...);
			std::cout << stream.str();
		}

		class PerformanceTestRender : public Renderer {
		private:
			int renderingDeviceCount;

			FrameBufferManager *front, *back;
			std::mutex swapLock;
			std::condition_variable swapCond;
			void maybeSwapBuffers() { trySwapBuffers(&front, &back, &swapLock, &swapCond); }

			FrameBuffer::BlockBank blockBank;
			bool updateScreenFromDevice, renderSingleIteration;


		protected:
			virtual bool setupSharedData(const Info &info, void *& sharedData) { return true; }

			virtual bool setupData(const Info &info, void *& data) {
				if (info.isGPU()) {
					FrameBuffer::DeviceBlockManager *manager = new FrameBuffer::DeviceBlockManager(
						info.device, FrameBuffer::DeviceBlockManager::CUDA_RENDER_STREAM_AUTO_SYNCH_ON_GET, 32);
					if (manager == NULL) {
						std::cout << "Failed to create a DeviceBlockManager" << std::endl;
						return false;
					}
					if (manager->errors() != 0) {
						delete manager;
						std::cout << "DeviceBlockManager has some errors" << std::endl;
						return false;
					}
					data = manager;
					return true;
				}
				else {
					data = NULL;
					return true;
				}
			}
			virtual bool prepareIteration() {
				blockBank.reset(*back->cpuHandle());
				return true;
			}
			virtual void iterateCPU(const Info &info) {
				FrameBuffer *buffer = back->cpuHandle();
				if (buffer == NULL) {
					std::cout << "CPU " << info.deviceThreadId << ": BUFFER IS NULL" << std::endl;
					return;
				}
				int start, end, blockSize, iter;
				iter = iteration();
				blockSize = buffer->getBlockSize();
				while (blockBank.getBlocks(4, &start, &end))
					for (int block = start; block < end; block++)
						for (int i = 0; i < blockSize; i++)
							colorPixel(block, blockSize, i, buffer, iter);
			}
			
			virtual void iterateGPU(const Info &info) {
				FrameBuffer *buffer = back->gpuHandle(info.device);
				if (buffer == NULL) { std::cout << "GPU " << info.device << ": BUFFER IS NULL" << std::endl; return; }
				FrameBuffer *cpuBuffer = back->cpuHandle();

				FrameBuffer::DeviceBlockManager *blockManager = info.getData<FrameBuffer::DeviceBlockManager>();
				if (!blockManager->setBuffers(cpuBuffer, buffer, &blockBank)) {
					std::cout << "GPU " << info.device << ": FAILED TO SET BUFFER" << std::endl; return; }

				int iter = iteration();
				bool synchNeeded = ((iter > 1) && ((renderingDeviceCount > 1) || (threadConfiguration().numHostThreads() > 0)));

				int start = 0, end = 0, blockSize = cpuBuffer->getBlockSize();
				while (blockManager->getBlocks(start, end, synchNeeded))
					renderBlocks<<<(end - start), blockSize, 0, blockManager->getRenderStream()>>>(start, buffer, iter);
				
				if (blockManager->errors() != 0) { std::cout << "GPU " << info.device << ": ERROR(S) OCCURED" << std::endl; return; }
				if (!blockManager->synchBlockSynchStream()) { std::cout << "GPU " << info.device << ": BLOCK SYNCH FAILED" << std::endl; return; }
			}

			virtual bool completeIteration() {
				cudaDeviceSynchronize();
				return true;
			}
			virtual bool clearData(const Info &info, void *& data) {
				if (data != NULL) {
					FrameBuffer::DeviceBlockManager *manager = ((FrameBuffer::DeviceBlockManager*)data);
					delete manager;
					data = NULL;
					return true;
				}
				return true;
			}
			virtual bool clearSharedData(const Info &info, void *& sharedData) { return true; }

		public:
			PerformanceTestRender(
				const Renderer::ThreadConfiguration &configuration,
				bool deviceWindow, bool singleIteration) : Renderer(configuration) {
				updateScreenFromDevice = deviceWindow;
				renderSingleIteration = singleIteration;
				renderingDeviceCount = threadConfiguration().numActiveDevices();
			}

			PerformanceTestRender& useBuffers(FrameBufferManager &frontBuffer, FrameBufferManager &backBuffer) {
				front = &frontBuffer;
				back = &backBuffer;
				return (*this);
			}

			void test() {
				std::thread windowThread;
				Windows::Window window;
				{
					volatile Count renderedFrames = 0, displayedFrames = 0;
					Count lastRenderedFrames = 0, lastDisplayedFrames = 0;

					volatile bool shouldStop = false;
					volatile bool stopped = false;

					windowThread = std::thread(
						frameBufferWindowThread,
						(const FrameBufferManager**)(renderSingleIteration ? (&front) : (&back)),
						(Windows::Window*)(&window),
						(std::mutex*)(&swapLock), (std::condition_variable*)(&swapCond),
						(volatile Count*)(&displayedFrames), &shouldStop, &stopped);

					clock_t lastTime = clock();

					while (true) {
						{
							if (window.dead() || shouldStop) {
								shouldStop = true;
								while (!stopped) swapCond.notify_all();
								break;
							}
						}
						{
							int windowWidth, windowHeight;
							if (!window.getDimensions(windowWidth, windowHeight)) continue;
							int imageWidth, imageHeight;
							back->cpuHandle()->getSize(&imageWidth, &imageHeight);
							if (imageWidth != windowWidth || imageHeight != windowHeight) {
								std::lock_guard<std::mutex> guard(swapLock);
								back->cpuHandle()->setResolution(windowWidth, windowHeight);
								back->makeDirty();
								resetIterations();
							}
						}
						{
							if (renderSingleIteration) resetIterations();
							if (!iterate()) {
								std::cout << "Error: iterate() failed..." << std::endl;
								shouldStop = true;
								continue;
							}
							renderedFrames++;
							if (renderSingleIteration) maybeSwapBuffers();
							else swapCond.notify_all();
						}
						{
							clock_t curTime = clock();
							float deltaTime = (((float)(curTime - lastTime)) / CLOCKS_PER_SEC);
							if (deltaTime >= 1.0f) {
								Count rendered = renderedFrames;
								Count displayed = displayedFrames;
								float fps = ((rendered - lastRenderedFrames) / deltaTime);
								float screenFps = ((displayed - lastDisplayedFrames) / deltaTime);
								lastRenderedFrames = rendered;
								lastDisplayedFrames = displayed;
								lastTime = curTime;
								std::cout << "FPS: " << fps << " (displayed: " << screenFps << ")" << std::endl;
							}
						}
					}
					windowThread.join();
				}
			}
		};
	}
	
	namespace Private {
		void testPerformance(FrameBufferManager &front, FrameBufferManager &back, Flags flags) {
			Renderer::ThreadConfiguration configuration;
			configuration.configureCPU(((flags & USE_CPU) != 0) ? Renderer::ThreadConfiguration::ALL : Renderer::ThreadConfiguration::NONE);
			if (((flags & USE_GPU) != 0) && (configuration.numDevices() > 0)) {
				if ((flags & TEST_SINGLE_GPU_ONLY) != 0) {
					configuration.configureEveryGPU(0);
					configuration.configureGPU(0, 2);
				}
				else configuration.configureEveryGPU(2);
			}
			else configuration.configureEveryGPU(0);
			PerformanceTestRender(
				configuration, 
				(flags & UPDATE_SCREEN_FROM_DEVICE) != 0, 
				(flags & TEST_FOR_SINGLE_ITERATION) != 0).useBuffers(front, back).test();
		}
	}
}

