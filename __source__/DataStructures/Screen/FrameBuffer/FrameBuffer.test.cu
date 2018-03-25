#include"FrameBuffer.test.cuh"
#include"../../Renderers/Renderer/Renderer.cuh"
#include"../../../Namespaces/Windows/Windows.h"
#include"../../../Namespaces/Device/Device.cuh"
#include<mutex>


namespace FrameBufferTest {
	namespace {
		typedef unsigned long long Count;

		void frameBufferWindowThread(
			const FrameBufferManager **buffer, Windows::Window *window, 
			std::mutex *bufferLock, std::condition_variable *bufferLockCond,
			volatile Count *displayedFrameCount) {
			while (true) {
				std::unique_lock<std::mutex> uniqueLock(*bufferLock);
				bufferLockCond->wait(uniqueLock);
				if (window->dead()) break;
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

		__device__ __host__ inline void colorPixel(int blockId, int pixelId, FrameBuffer *buffer, int iteration) {
#ifdef __CUDA_ARCH__
			Color color(0, 1, 0, 1);
#else
			Color color(0, 0, 1, 1);
#endif
			if (iteration > 1) buffer->blendBlockPixelColor(blockId, pixelId, color, (1.0f / ((float)iteration)));
			else buffer->setBlockPixelColor(blockId, pixelId, color);
		}

		__global__ static void renderBlocks(int startBlock, FrameBuffer *buffer, int iteration) {
			colorPixel(startBlock + blockIdx.x, threadIdx.x, buffer, iteration);
		}

		static int multiprocessorCount() { return Device::multiprocessorCount(); }

		class PerformanceTestRender : public Renderer {
		private:
			FrameBufferManager *front, *back;
			std::mutex swapLock;
			std::condition_variable swapCond;
			void maybeSwapBuffers() { trySwapBuffers(&front, &back, &swapLock, &swapCond); }

			FrameBuffer::BlockBank blockBank;


		protected:
			virtual bool setupSharedData(const Info &info, void *& sharedData) {
				if (info.isGPU()) {
					if (cudaSetDevice(info.device) != cudaSuccess) return false;
					sharedData = ((void*)new int(multiprocessorCount()));
					return (sharedData != NULL);
				}
				else {
					sharedData = NULL;
					return true;
				}
			}
			virtual bool setupData(const Info &info, void *& data) {
				if (info.isGPU()) {
					if (cudaSetDevice(info.device) != cudaSuccess) return false;
					cudaStream_t *stream = new cudaStream_t();
					if (stream == NULL) return false;
					if (cudaStreamCreate(stream) != cudaSuccess) {
						delete stream;
						data = NULL;
						std::cout << "GPU " << info.device << " FAILED TO ALLOCATE STREAM..." << std::endl;
						return false;
					}
					data = (void*)stream;
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
				while (blockBank.getBlocks(1, &start, &end))
					for (int i = 0; i < blockSize; i++)
						colorPixel(start, i, buffer, iter);
			}
			virtual void iterateGPU(const Info &info) {
				FrameBuffer *buffer = back->gpuHandle(info.device);
				if (buffer == NULL) {
					std::cout << "GPU " << info.device << ": BUFFER IS NULL" << std::endl;
					return;
				}

				int iter = iteration();

				if (iter > 1) back->cpuHandle()->updateDeviceInstance(buffer);

				cudaStream_t &stream = (*((cudaStream_t*)info.data));

				int start, end, blockSize, devBlockCount;
				blockSize = back->cpuHandle()->getBlockSize();
				
				devBlockCount = ((*((int*)info.sharedData)) * 4);
				
				while (blockBank.getBlocks(devBlockCount, &start, &end)) {
					renderBlocks << <(end - start), blockSize, 0, stream >> > (start, buffer, iter);
				}
				
				if (cudaStreamSynchronize(stream) != cudaSuccess)
					std::cout << "GPU " << info.device << " FAILED TO SYNCHRONIZE THE THREAD..." << std::endl;

				// TODO: Synch cpu instance...
			}
			virtual bool completeIteration() {
				cudaDeviceSynchronize();
				return true;
			}
			virtual bool clearData(const Info &info, void *& data) {
				if (data != NULL) {
					cudaStream_t *stream = ((cudaStream_t*)data);
					bool success = (cudaStreamDestroy(*stream) == cudaSuccess);
					delete stream;
					data = NULL;
					return success;
				}
				return true;
			}
			virtual bool clearSharedData(const Info &info, void *& sharedData) {
				if (sharedData != NULL) {
					delete ((int*)sharedData);
					sharedData = NULL;
				}
				return true;
			}

		public:
			PerformanceTestRender(const Renderer::ThreadConfiguration &configuration) : Renderer(configuration) {}

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
					
					windowThread = std::thread(
						frameBufferWindowThread,
						(const FrameBufferManager**)(&front), (Windows::Window*)(&window),
						(std::mutex*)(&swapLock), (std::condition_variable*)(&swapCond),
						(volatile Count*)(&displayedFrames));

					clock_t lastTime = clock();

					while (!window.dead()) {
						{
							int windowWidth, windowHeight;
							if (!window.getDimensions(windowWidth, windowHeight)) break;
							int imageWidth, imageHeight;
							back->cpuHandle()->getSize(&imageWidth, &imageHeight);
							if (imageWidth != windowWidth || imageHeight != windowHeight) {
								back->cpuHandle()->setResolution(windowWidth, windowHeight);
								back->makeDirty();
								resetIterations();
							}
						}
						{
							//resetIterations();
							if (!iterate()) {
								std::cout << "Error: iterate() failed..." << std::endl;
								break;
							}
							renderedFrames++;
							maybeSwapBuffers();
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
				}
				swapLock.lock();
				swapCond.notify_all();
				swapLock.unlock();
				windowThread.join();
			}
		};
	}
	
	namespace Private {
		void testPerformance(FrameBufferManager &front, FrameBufferManager &back, Flags flags) {
			Renderer::ThreadConfiguration configuration;
			configuration.configureCPU(((flags & USE_CPU) != 0) ? Renderer::ThreadConfiguration::ALL : Renderer::ThreadConfiguration::NONE);
			configuration.configureEveryGPU((flags & USE_GPU) ? 1 : 0);
			PerformanceTestRender(configuration).useBuffers(front, back).test();
		}
	}
}

