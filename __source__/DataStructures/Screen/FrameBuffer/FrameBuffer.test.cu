#include"FrameBuffer.test.cuh"
#include"../../Renderers/Renderer/Renderer.cuh"
#include"../../../Namespaces/Windows/Windows.h"
#include<mutex>


namespace FrameBufferTest {
	namespace {
		typedef unsigned long long Count;

		void windowUpdateThread(
			const FrameBufferManager **buffer, Windows::Window *window, 
			std::mutex *bufferLock, std::condition_variable *bufferLockCond,
			Count *displayedFrameCount) {
			while (true) {
				std::unique_lock<std::mutex> uniqueLock(*bufferLock);
				bufferLockCond->wait(uniqueLock);
				if (window->dead()) break;
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
				bufferLock->unlock();
				swapCond->notify_all();
			}
		}

		class PerformanceTestRender : public Renderer {
		private:
			FrameBufferManager *front, *back;
			std::mutex swapLock;
			std::condition_variable swapCond;
			void maybeSwapBuffers() { trySwapBuffers(&front, &back, &swapLock, &swapCond); }


		protected:
			virtual bool setupSharedData(const Info &info, void *& sharedData) {
				// __TODO__
				return true;
			}
			virtual bool setupData(const Info &info, void *& data) {
				// __TODO__
				return true;
			}
			virtual bool prepareIteration() {
				// __TODO__
				return true;
			}
			virtual void iterateCPU(const Info &info) {

			}
			virtual void iterateGPU(const Info &info) {

			}
			virtual bool completeIteration() {
				// __TODO__
				return true;
			}
			virtual bool clearData(const Info &info, void *& data) {
				// __TODO__
				return true;
			}
			virtual bool clearSharedData(const Info &info, void *& sharedData) {
				// __TODO__
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
				{
					Windows::Window window;
					Count renderedFrames = 0, displayedFrames = 0;
					Count lastRenderedFrames = 0, lastDisplayedFrames = 0;
					clock_t lastTime = clock();
					//windowThread = std::thread(
					//	windowUpdateThread,
					//	&front, &window,
					//	&swapLock, &swapCond,
					//	&displayedFrames);
					while (!window.dead()) {
						resetIterations();
						if (!iterate()) {
							std::cout << "Error: iterate() failed..." << std::endl;
							break;
						}
						renderedFrames++;
						maybeSwapBuffers();

					}
				}
				swapCond.notify_all();
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

