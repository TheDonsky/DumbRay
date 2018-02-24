#include "MemoryMappedFrameBuffer.test.cuh"
#include "MemoryMappedFrameBuffer.cuh"
#include "../FrameBufferManager/FrameBufferManager.cuh"
#include "../../../Renderers/Renderer/Renderer.cuh"
#include "../../../../Namespaces/Tests/Tests.h"
#include "../../../../Namespaces/Windows/Windows.h"
#include "../../../Objects/Scene/SceneHandler/SceneHandler.cuh"
#include "../../../../Namespaces/Device/Device.cuh"
#include <thread>
#include <mutex>
#include <iostream>

namespace MemoryMappedFrameBufferTest {
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 8
#define BLOCK_SIZE (BLOCK_WIDTH * BLOCK_HEIGHT)
#define BLOCKS_ALONG_AXIS(size, chunkSize) ((size + chunkSize - 1) / chunkSize)
#define WIDTH_BLOCKS(w) BLOCKS_ALONG_AXIS(w, BLOCK_WIDTH)
#define HEIGHT_BLOCKS(h) BLOCKS_ALONG_AXIS(h, BLOCK_HEIGHT)
#define GPU_BLOCKS_PER_KERNEL_BLOCK 4
#define GPU_KERNEL_BLOCKS_PER_SM 8
#define WANTED_BLOCKS(sm) (sm * GPU_BLOCKS_PER_KERNEL_BLOCK * GPU_KERNEL_BLOCKS_PER_SM)
	namespace {
		bool smCount(int &result) {
			result = Device::multiprocessorCount();
			if (result > 0) return true;
			else {
				result = 0;
				return false;
			}
		}

		template<typename First, typename... Others> 
		void print(const First &first, const Others&... others);
		template<typename Type>
		void print(const Type &value) { std::cout << value; }
		template<typename First, typename... Others>
		void print(const First &first, const Others&... others) { std::cout << first; print(others...); }
		static std::mutex printGuard;
		template<typename... Args>
		void log(const Args&... args) { printGuard.lock(); print(args...); printGuard.unlock(); }


		__device__ __host__ inline bool getPixel(int width, int height, int blockId, int pixelId, int &x, int &y) {
			register int widthBlocks = WIDTH_BLOCKS(width);
			register int blockY = (blockId / widthBlocks);
			register int blockX = (blockId - (blockY * widthBlocks));
			register int yOffset = (pixelId / BLOCK_WIDTH);
			register int xOffset = (pixelId - (yOffset * BLOCK_WIDTH));
			register int posX = ((blockX * BLOCK_WIDTH) + xOffset);
			register int posY = ((blockY * BLOCK_HEIGHT) + yOffset);
			if ((posX < width) && (posY < height)) {
				x = posX;
				y = posY;
				return true;
			}
			else return false;
		}

		__device__ __host__ inline void renderPixel(FrameBuffer *buffer, int width, int height, int blockId, int pixelId, int frame, float blendingAmount) {
			int x, y;
			if (getPixel(width, height, blockId, pixelId, x, y)) {
				Color pixelColor;
#ifndef __CUDA_ARCH__
				pixelColor(
					(255 & ((x ^ y) + frame)) / 255.0f,
					(255 & ((x & y) - frame)) / 255.0f,
					(255 & ((x | y) + frame)) / 255.0f);
#else
				pixelColor(
					((x + frame) & 255) / 255.0f, 
					((y - frame) & 255) / 255.0f, 
					(((x + y) ^ frame) & 255) / 255.0f);
#endif // !__CUDA_ARCH__
				if (blendingAmount >= 1.0f) buffer->setColor(x, y, pixelColor);
				else buffer->blendColor(x, y, pixelColor, blendingAmount);
			}
		}

		__global__ static void renderPixels(
			FrameBuffer *buffer, int width, int height,
			int startBlockId, int endBlockId, int frame, float blendingAmount) {
			register int block = (startBlockId + (blockIdx.x * GPU_BLOCKS_PER_KERNEL_BLOCK));
			register int endBlock = (block + GPU_BLOCKS_PER_KERNEL_BLOCK);
			register int pixelId = (threadIdx.x / GPU_BLOCKS_PER_KERNEL_BLOCK);
			
			//Color *color = buffer->getData();

			if (endBlock > endBlockId) endBlock = endBlockId;
			while (block < endBlock) {
				//*
				renderPixel(buffer, width, height, block, pixelId, frame, blendingAmount);
				/*/
				register int x, y;
				if (getPixel(width, height, block, pixelId, x, y))
					color[(width * y) + x] = Color(0, 1, 0);
				//*/
				block++;
			}
		}

		class TestRender : public Renderer {
		private:
			FrameBufferManager *buffer;
			struct Bank {
				int count;
				std::mutex lock;
				void set(int num) { count = num; }
				bool get(int num, int &start, int &end) {
					lock.lock();
					bool rv;
					if (count <= 0) rv = false;
					else {
						end = count;
						count -= num;
						if (count < 0) count = 0;
						start = count;
						rv = true;
					}
					lock.unlock();
					return rv;
				}
			};
			Bank bank;

			Scene<BakedTriFace> scene;
			SceneHandler<BakedTriFace> sceneHandler;

			int frameId;

			uint32_t params;


		public:
			enum Flags {
				FLAG_NONE = 0,
				FLAG_SINGLE_ITERATION = 1,
				FLAG_RENDER_SCENE = 2
			};

			TestRender(const Renderer::ThreadConfiguration &configuration, uint32_t flags = FLAG_SINGLE_ITERATION)
				: Renderer(configuration), sceneHandler(scene) { 
				buffer = NULL;
				params = flags;
			}
			virtual ~TestRender() { killRenderThreads(); }
			void setBuffer(FrameBufferManager *buff) {
				buffer = buff;
				resetIterations();
			}
			void setFrame(int frame) {
				frameId = (frame / 8);
				resetIterations();
			}

		private:
			struct GPUcontext {
				enum Flags {
					BLOCK_COUNT_FAILED = 1,
					STREAM_NOT_INITIALIZED = 2,
					CONTEXT_FAILED = 4
				};
				uint8_t flags;
				int blocks;
				cudaStream_t stream;

				inline GPUcontext() {
					flags = 0;
					if (smCount(blocks)) blocks *= GPU_KERNEL_BLOCKS_PER_SM;
					else flags |= BLOCK_COUNT_FAILED;
					if (cudaStreamCreate(&stream) != cudaSuccess) flags |= STREAM_NOT_INITIALIZED;
					if ((flags & (BLOCK_COUNT_FAILED | STREAM_NOT_INITIALIZED)) != 0) flags |= CONTEXT_FAILED;
				}
				inline ~GPUcontext() {
					if ((flags & STREAM_NOT_INITIALIZED) != 0)
						cudaStreamDestroy(stream);
				}
				inline bool hasFailed()const {
					return ((flags & CONTEXT_FAILED) != 0);
				}
				inline void registerError() {
					flags |= CONTEXT_FAILED;
				}
			};

		protected:
			virtual bool setupSharedData(const Info &info, void *& sharedData) { 
				if (info.isGPU())
					if (!sceneHandler.selectGPU(info.device))
						return false;
				return true;
			}
			virtual bool setupData(const Info &info, void *& data) { 
				if (!info.isGPU()) return true;
				if (!sceneHandler.uploadToGPU(info.device)) {
					log("Upload failed for device ", info.device, "\n"); 
					return false;
				}
				GPUcontext *context = new GPUcontext();
				if (context != NULL) {
					if (!context->hasFailed()) {
						data = ((void*)context);
						return true;
					}
					else delete context;
				}
				data = NULL;
				return false;
			}
			virtual bool prepareIteration() { 
				if (buffer != NULL) {
					int width, height;
					buffer->cpuHandle()->getSize(&width, &height);
					if (width <= 0 || height <= 0) {
						log("ERROR: Buffer size invalid\n");
						return false;
					}
					int blockCount = (WIDTH_BLOCKS(width) * HEIGHT_BLOCKS(height));
					bank.set(blockCount);
					return true;
				}
				else {
					log("ERROR: No buffer set\n");
					return false;
				}
			}
			virtual void iterateCPU(const Info &info) {
				int width, height;
				FrameBuffer *frame = buffer->cpuHandle();
				if (frame == NULL) return;
				frame->getSize(&width, &height);
				int pointer, end;
				float blending = (((params & FLAG_SINGLE_ITERATION) == 0) ? (1.0f / ((float)iteration())) : 1.0f);
				while (bank.get(1, pointer, end)) {
					while (pointer < end) {
						for (int i = 0; i < BLOCK_SIZE; i++)
							renderPixel(frame, width, height, pointer, i, frameId, blending);
						pointer++;
					}
				}
			}
			virtual void iterateGPU(const Info &info) {
				GPUcontext *context = ((GPUcontext*)info.data);
				Scene<BakedTriFace> *GPUscene = sceneHandler.getHandleGPU(info.device);
				
				int width, height;
				FrameBuffer *frame = buffer->cpuHandle();
				frame->getSize(&width, &height);
				FrameBuffer *GPUframe = buffer->gpuHandle(info.device);

				if (GPUscene == NULL || GPUframe == NULL || GPUframe == NULL) return;

				float blending = (((params & FLAG_SINGLE_ITERATION) == 0) ? (1.0f / ((float)iteration())) : 1.0f);

				const int blockCount = (context->blocks * GPU_BLOCKS_PER_KERNEL_BLOCK);
				int start, end;
				const int blk = context->blocks;
				const int thr = (GPU_BLOCKS_PER_KERNEL_BLOCK * BLOCK_SIZE);
				int prevStart = -1;
				int prevEnd = -1;
				while (bank.get(blockCount, start, end)) {
					if (prevStart < prevEnd) {
						if (cudaStreamSynchronize(context->stream) != cudaSuccess) { log("KERNEL FAILURE....\n"); return; }
						else if (!frame->updateBlocks(start, end, GPUframe)) { log("UPDATE FAILURE.....\n"); return; }
					}
					renderPixels<<<blk, thr, 0, context->stream>>>(GPUframe, width, height, start, end, frameId, blending);
					prevStart = start;
					prevEnd = end;
				}
				if (prevStart < prevEnd) {
					if (cudaStreamSynchronize(context->stream) != cudaSuccess) { log("KERNEL FAILURE....\n"); return; }
					else if (!frame->updateBlocks(start, end, GPUframe)) { log("UPDATE FAILURE.....\n"); return; }
				}
			}
			virtual bool completeIteration() { return true; }
			virtual bool clearData(const Info &info, void *& data) {
				if (!info.isGPU()) return true;
				if (data != NULL) {
					GPUcontext *context = ((GPUcontext*)data);
					delete context;
					data = NULL;
				}
				return true; 
			}
			virtual bool clearSharedData(const Info &info, void *& sharedData) { return true; }
		};


		typedef Windows::Window Window;

		inline static void windowUpdateThread(Window *window, 
			std::mutex *lock, std::condition_variable *condition, 
			FrameBufferManager **buffer, int *framesDisplayed, 
			bool useMemcpy) {
			cudaStream_t stream;
			int surface = 1; Color *color;
			if (useMemcpy) {
				if (cudaStreamCreate(&stream) != cudaSuccess) { log("STREAM CREATION FAILED (windowUpdateThread)\n"); return; }
				color = new Color[surface];
				if (color == NULL) { log("ALLOCATION ERROR (windowUpdateThread)\n"); return; }
			}
			while (!window->dead()) {
				std::unique_lock<std::mutex> uniqueLock(*lock);
				condition->wait(uniqueLock);
				const FrameBuffer &frameBuffer = (*(*buffer)->cpuHandle());
				int width, height;
				frameBuffer.getSize(&width, &height);
				if (width > 0 && height > 0) {
					if (useMemcpy) {
						int newSurface = (width * height);
						if (surface < newSurface) {
							delete[] color; surface = newSurface; color = new Color[surface];
							if (color == NULL) { log("ALLOCATION ERROR (windowUpdateThread)\n"); break; }
						}
						//if (cudaMemcpyAsync(color, frameBuffer.getData(), sizeof(Color) * newSurface, cudaMemcpyDefault, stream) != cudaSuccess) {
						//	log("cudaMemcpyError (windowUpdateThread)\n"); break;
						//}
						if (cudaStreamSynchronize(stream) != cudaSuccess) { log("STREAM SYNCHRONISATION FAILED (windowUpdateThread)\n"); break; }
						window->updateFrameHost(color, width, height);
					}
					//else window->updateFrameHost(frameBuffer.getData(), width, height);
					(*framesDisplayed)++;
				}
			}
			if (useMemcpy) {
				if (color != NULL) delete[] color;
				if (cudaStreamDestroy(stream) != cudaSuccess) log("FAILED TO DESTROY STREAM (windowUpdateThread)\n");
			}
		}

		struct Size { int width, height; };

		inline static void setResolution(FrameBuffer &buffer, void *aux) {
			Size &size = (*((Size*)aux));
			buffer.setResolution(size.width, size.height);
		}

		inline static void run() {
			Window window("MemoryMappedFrameBuffer TEST WINDOW");
			FrameBufferManager a, b;
			a.cpuHandle()->use<MemoryMappedFrameBuffer>();
			b.cpuHandle()->use<MemoryMappedFrameBuffer>();
			FrameBufferManager *front = (&a);
			FrameBufferManager *back = (&b);
			std::mutex lock;
			std::condition_variable condition;
			TestRender *renderer = new TestRender(Renderer::ThreadConfiguration::gpuOnly());
			int framesDisplayed = 0;
			int framesDisplayedLast = 0;
			std::thread windowUpdate(windowUpdateThread, &window, &lock, &condition, &front, &framesDisplayed, false);
			long long lastTime = clock();
			double iterationCount = 0;
			int frameId = 0;
			while (!window.dead()) {
				renderer->setFrame(frameId);
				renderer->iterate();
				frameId++;
				if (lock.try_lock()) {
					FrameBufferManager *tmp = front;
					front = back;
					back = tmp;
					Size size;
					window.getDimensions(size.width, size.height);
					back->edit(setResolution, &size);
					renderer->setBuffer(back);
					lock.unlock();
					condition.notify_all();
				}
				iterationCount++;
				long long curTime = clock();
				long long deltaTime = (curTime - lastTime);
				if (deltaTime >= CLOCKS_PER_SEC) {
					double elapsedTime = (((double)deltaTime) / ((double)CLOCKS_PER_SEC));
					double iterationsPerSecond = (iterationCount / elapsedTime);
					double averageIterationTime = (elapsedTime / iterationCount);
					int width, height;
					back->cpuHandle()->getSize(&width, &height);
					int shownFrames = (framesDisplayed - framesDisplayedLast);
					framesDisplayedLast = framesDisplayed;
					double shownFramesPerSecond = (((double)shownFrames) / elapsedTime);
					log("ITER PER SECOND: ", iterationsPerSecond,
						"; ITER TIME: ", averageIterationTime,
						"; SHOWN FRAMES: ", shownFrames,
						"; SHOWN FRAMES PER SEC: ", shownFramesPerSecond,
						"; RESOLUTION: ", width, " * ", height, '\n');
					iterationCount = 0;
					lastTime = curTime;
				}
			}
			delete renderer;
			windowUpdate.join();
		}
	}
	void test() {
		Tests::runTest(run, "Testing MemoryMappedFrameBuffer (turn off the window to quit)");
	}
}
