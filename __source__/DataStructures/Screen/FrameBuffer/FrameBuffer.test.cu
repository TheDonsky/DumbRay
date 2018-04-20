#include"FrameBuffer.test.cuh"
#include"../../Renderers/Renderer/Renderer.cuh"
#include"../BufferedWindow/BufferedWindow.cuh"
#include"../../../Namespaces/Device/Device.cuh"
#include"../../Primitives/Compound/Pair/Pair.cuh"
#include"../../Renderers/BufferedRenderProcess/BufferedRenderProcess.cuh"
#include<mutex>
#include<sstream>


namespace FrameBufferTest {
	namespace {
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

		typedef unsigned long long Count;

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

		class PerformanceTestRender : public BufferedRenderer {
		private:
			int renderingDeviceCount;
			bool hostBlockSynchNeeded;

			FrameBuffer::BlockBank blockBank;
			bool updateScreenFromDevice, renderSingleIteration;


		protected:
			virtual bool setupSharedData(const Info &info, void *& sharedData) { return true; }

			virtual bool setupData(const Info &info, void *& data) {
				if (info.isGPU()) {
					FrameBuffer::DeviceBlockManager *manager = new FrameBuffer::DeviceBlockManager(
						info.device, (hostBlockSynchNeeded ?
							FrameBuffer::DeviceBlockManager::CUDA_RENDER_STREAM_AUTO_SYNCH_ON_GET :
							FrameBuffer::DeviceBlockManager::CUDA_MANUALLY_SYNCH_HOST_BLOCKS),
						hostBlockSynchNeeded ? 16 : 2048);
					if (manager == NULL) { print("Failed to create a DeviceBlockManager\n"); return false; }
					if (manager->errors() != 0) {
						delete manager;
						print("DeviceBlockManager has some errors\n");
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
				blockBank.reset(*getFrameBuffer()->cpuHandle());
				return true;
			}
			virtual void iterateCPU(const Info &info) {
				FrameBuffer *buffer = getFrameBuffer()->cpuHandle();
				if (buffer == NULL) { print("CPU ", info.deviceThreadId, ": BUFFER IS NULL\n"); return; }
				int start, end, blockSize, iter;
				iter = iteration();
				blockSize = buffer->getBlockSize();
				while (blockBank.getBlocks(4, &start, &end))
					for (int block = start; block < end; block++)
						for (int i = 0; i < blockSize; i++)
							colorPixel(block, blockSize, i, buffer, iter);
			}

			virtual void iterateGPU(const Info &info) {
				FrameBuffer *buffer = getFrameBuffer()->gpuHandle(info.device);
				if (buffer == NULL) { print("GPU ", info.device, ": BUFFER IS NULL\n"); return; }
				FrameBuffer *cpuBuffer = getFrameBuffer()->cpuHandle();

				FrameBuffer::DeviceBlockManager *blockManager = info.getData<FrameBuffer::DeviceBlockManager>();
				if (!blockManager->setBuffers(cpuBuffer, buffer, &blockBank)) {
					print("GPU ", info.device, ": FAILED TO SET BUFFER\n"); return;
				}

				int iter = iteration();
				bool synchNeeded = ((iter > 1) && hostBlockSynchNeeded);

				int start = 0, end = 0, blockSize = cpuBuffer->getBlockSize();
				while (blockManager->getBlocks(start, end, synchNeeded))
					renderBlocks << <(end - start), blockSize, 0, blockManager->getRenderStream() >> >(start, buffer, iter);

				if (blockManager->errors() != 0) { print("GPU ", info.device, ": ERROR(S) OCCURED\n"); return; }
				if (hostBlockSynchNeeded) if (!blockManager->synchBlockSynchStream()) { print("GPU ", info.device, ": BLOCK SYNCH FAILED\n"); return; }
				else if (!blockManager->synchRenderStream()) { print("GPU ", info.device, ": RENDER STREAM SYNCH FAILED\n"); return; }
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
				bool deviceWindow, bool singleIteration) : BufferedRenderer(configuration) {
				updateScreenFromDevice = deviceWindow;
				renderSingleIteration = singleIteration;
				renderingDeviceCount = threadConfiguration().numActiveDevices();
				hostBlockSynchNeeded = ((renderingDeviceCount > 1) || (threadConfiguration().numHostThreads() > 0));
			}
			~PerformanceTestRender() { killRenderThreads(); }


		private:
			volatile Count renderedFrames;
			volatile Count lastRenderedFrames, lastDisplayedFrames;
			volatile clock_t lastTime;
			BufferedWindow *displayWindow;

		public:


			static void iterationCompletionCallback(void *testCase) {
				PerformanceTestRender *self = ((PerformanceTestRender*)testCase);
				self->renderedFrames++;
				clock_t curTime = clock();
				float deltaTime = (((float)(curTime - self->lastTime)) / CLOCKS_PER_SEC);
				if (deltaTime >= 1.0f) {
					Count rendered = self->renderedFrames;
					Count displayed = self->displayWindow->framesDisplayed();
					float fps = ((rendered - self->lastRenderedFrames) / deltaTime);
					float screenFps = ((displayed - self->lastDisplayedFrames) / deltaTime);
					self->lastRenderedFrames = rendered;
					self->lastDisplayedFrames = displayed;
					self->lastTime = curTime;
					std::cout << "FPS: " << fps << " (displayed: " << screenFps << ")" << std::endl;
				}
			}

			static void printErrorOnIterate(void*) {
			}

			void test(FrameBufferManager &frontBuffer, FrameBufferManager &backBuffer) {
				BufferedWindow bufferedWindow(hostBlockSynchNeeded ? 0 : BufferedWindow::SYNCH_FRAME_BUFFER_FROM_DEVICE);
				BufferedRenderProcess bufferedRenderProcess;

				bufferedRenderProcess.setRenderer(this);
				if (renderSingleIteration) bufferedRenderProcess.setDoubleBuffers(&backBuffer, &frontBuffer);
				else bufferedRenderProcess.setBuffer(&backBuffer);
				bufferedRenderProcess.setInfinateTargetIterations();
				bufferedRenderProcess.setTargetDisplayWindow(&bufferedWindow);
				bufferedRenderProcess.setTargetResolutionToWindowSize();

				renderedFrames = 0;
				lastRenderedFrames = lastDisplayedFrames = 0;
				displayWindow = &bufferedWindow;
				lastTime = clock();
				bufferedRenderProcess.setIterationCompletionCallback(iterationCompletionCallback, this);

				bufferedRenderProcess.start();
				while (!bufferedWindow.windowClosed()) std::this_thread::sleep_for(std::chrono::milliseconds(32));
				bufferedRenderProcess.end();
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
				(flags & TEST_FOR_SINGLE_ITERATION) != 0).test(front, back);
		}
	}
}

