#include"FrameBuffer.test.cuh"
#include"../../Renderers/Renderer/Renderer.cuh"
#include"../BufferedWindow/BufferedWindow.cuh"
#include"../../../Namespaces/Device/Device.cuh"
#include"../../Primitives/Compound/Pair/Pair.cuh"
#include"../../Renderers/BufferedRenderProcess/BufferedRenderProcess.cuh"
#include"../../Renderers/BlockRenderer/BlockRenderer.cuh"
#include"../../Renderers/BufferedRenderProcess/BufferedRenderProcess.test.cuh"
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
				const Renderer::ThreadConfiguration &configuration) : BlockRenderer(configuration) {
			}
			~PerformanceTestRender() { killRenderThreads(); }
		};
	}

	namespace Private {
		void testPerformance(const std::string& bufferClassName, FrameBufferManager &front, FrameBufferManager &back) {
			std::cout << "__________________________________________________" << std::endl;
			std::cout << "TESTING PERFORMANCE OF " << bufferClassName << ": " << std::endl;
			BufferedRenderProcessTest::runTestGauntlet<PerformanceTestRender>(&front, &back);
		}
	}
}

