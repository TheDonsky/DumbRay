#include"FrameBuffer.test.cuh"
#include"../../Renderers/Renderer/Renderer.cuh"
#include"../BufferedWindow/BufferedWindow.cuh"
#include"../../../Namespaces/Device/Device.cuh"
#include"../../Primitives/Compound/Pair/Pair.cuh"
#include"../../Renderers/BufferedRenderProcess/BufferedRenderProcess.cuh"
#include"../../Renderers/BlockRenderer/BlockRenderer.cuh"
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

		class PerformanceTestRender : public BlockRenderer {
		private:
			bool renderSingleIteration;
			volatile Count renderedFrames;
			volatile Count lastRenderedFrames, lastDisplayedFrames;
			volatile clock_t lastTime;
			BufferedWindow *displayWindow;

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

		protected:
			virtual bool renderBlocksCPU(const Info &, FrameBuffer *buffer, int startBlock, int endBlock) {
				int blockSize = buffer->getBlockSize();
				int iter = iteration();
				for (int block = startBlock; block < endBlock; block++)
					for (int i = 0; i < blockSize; i++)
						colorPixel(block, blockSize, i, buffer, iter);
				return true;
			}
			virtual bool renderBlocksGPU(const Info &, FrameBuffer *host, FrameBuffer *device, int startBlock, int endBlock, cudaStream_t &renderStream) {
				renderBlocks<<<(endBlock - startBlock), host->getBlockSize(), 0, renderStream>>>(startBlock, device, iteration());
				return true;
			}

		public:
			PerformanceTestRender(
				const Renderer::ThreadConfiguration &configuration, bool singleIteration) : BlockRenderer(configuration) {
				renderSingleIteration = singleIteration;
			}
			~PerformanceTestRender() { killRenderThreads(); }


		public:
			void test(FrameBufferManager &frontBuffer, FrameBufferManager &backBuffer) {
				BufferedWindow bufferedWindow(automaticallySynchesHostBlocks() ? 0 : BufferedWindow::SYNCH_FRAME_BUFFER_FROM_DEVICE);
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
			PerformanceTestRender(configuration, (flags & TEST_FOR_SINGLE_ITERATION) != 0).test(front, back);
		}
	}
}

