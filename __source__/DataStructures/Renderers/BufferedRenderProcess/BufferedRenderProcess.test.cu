#include "BufferedRenderProcess.test.cuh"
#include "../../../Namespaces/Tests/Tests.h"
#include <string>

namespace BufferedRenderProcessTest {
	void FramerateLogger::start() {
		renderedFrames = 0;
		lastRenderedFrames = lastDisplayedFrames = 0;
		lastTime = clock();
	}
	void FramerateLogger::setBufferedWindow(BufferedWindow *bufferedWindow) {
		displayWindow = bufferedWindow;
	}
	void FramerateLogger::registerIterationCompletionCallback(BufferedRenderProcess *process) {
		process->setIterationCompletionCallback(iterationCompletionCallback, (void*)this);
		
	}

	void FramerateLogger::iterationCompletionCallback(void *testCase) {
		FramerateLogger *self = ((FramerateLogger*)testCase);
		self->renderedFrames++;
		clock_t curTime = clock();
		float deltaTime = (((float)(curTime - self->lastTime)) / CLOCKS_PER_SEC);
		if (deltaTime >= 1.0f) {
			uint32_t rendered = self->renderedFrames;
			uint32_t displayed = ((self->displayWindow == NULL) ? 0 : (uint32_t)self->displayWindow->framesDisplayed());
			float fps = ((rendered - self->lastRenderedFrames) / deltaTime);
			float screenFps = ((displayed - self->lastDisplayedFrames) / deltaTime);
			self->lastRenderedFrames = rendered;
			self->lastDisplayedFrames = displayed;
			self->lastTime = curTime;
			std::cout << "FPS: " << fps << " (displayed: " << screenFps << ")" << std::endl;
		}
	}
	namespace {
		static void testBufferedRenderProcess(
			BufferedRenderer*(*bufferedRendererCreateFunction)(const Renderer::ThreadConfiguration &configuration, void *aux),
			void *createFnAux, const Renderer::ThreadConfiguration &configuration, BufferedRenderProcess *process) {

			BufferedRenderer *renderer = bufferedRendererCreateFunction(configuration, createFnAux);
			BufferedWindow bufferedWindow(
				((configuration.numHostThreads() > 0) || (configuration.numActiveDevices() > 1)) ? 
				0 : BufferedWindow::SYNCH_FRAME_BUFFER_FROM_DEVICE);
			process->setRenderer(renderer);
			process->setTargetDisplayWindow(&bufferedWindow);
			FramerateLogger logger;
			logger.setBufferedWindow(&bufferedWindow);
			logger.registerIterationCompletionCallback(process);
			logger.start();
			process->start();
			while (!bufferedWindow.windowClosed()) std::this_thread::sleep_for(std::chrono::milliseconds(32));
			process->end();
			delete renderer;
		}
	}

	void runTestGauntlet(
		BufferedRenderer*(*bufferedRendererCreateFunction)(const Renderer::ThreadConfiguration &configuration, void *aux),
		void *createFnAux, FrameBufferManager *bufferA, FrameBufferManager *bufferB, uint32_t tests) {

		BufferedRenderProcess bufferedRenderProcess;
		bufferedRenderProcess.setInfinateTargetIterations();
		bufferedRenderProcess.setTargetResolutionToWindowSize();

		Renderer::ThreadConfiguration configuration;

		bufferedRenderProcess.setBuffer(bufferA);
		if ((tests & TEST_MULTI_ITER_CPU_ONLY) != NULL) {
			configuration.configureCPU(Renderer::ThreadConfiguration::ALL);
			configuration.configureEveryGPU(Renderer::ThreadConfiguration::NONE);
			Tests::runTest(testBufferedRenderProcess, "TEST_MULTI_ITER_CPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
		}
		if (configuration.numDevices() > 0) {
			if ((tests & TEST_MULTI_ITER_GPU_ONLY) != NULL) {
				configuration.configureCPU(Renderer::ThreadConfiguration::NONE);
				configuration.configureEveryGPU(2);
				Tests::runTest(testBufferedRenderProcess, "TEST_MULTI_ITER_GPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
			}
			if ((tests & TEST_MULTI_ITER_CPU_AND_GPU) != NULL) {
				configuration.configureCPU(Renderer::ThreadConfiguration::ALL);
				configuration.configureEveryGPU(2);
				Tests::runTest(testBufferedRenderProcess, "TEST_MULTI_ITER_CPU_AND_GPU", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
			}
		}
		if ((tests & TEST_MULTI_ITER_1_CPU_ONLY) != NULL) {
			configuration.configureCPU(Renderer::ThreadConfiguration::ONE);
			configuration.configureEveryGPU(Renderer::ThreadConfiguration::NONE);
			Tests::runTest(testBufferedRenderProcess, "TEST_MULTI_ITER_1_CPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
		}
		if (((tests & TEST_MULTI_ITER_1_GPU_ONLY) != NULL) && (configuration.numDevices() > 1)) {
			configuration.configureCPU(Renderer::ThreadConfiguration::NONE);
			configuration.configureEveryGPU(0);
			configuration.configureGPU(0, 2);
			Tests::runTest(testBufferedRenderProcess, "TEST_MULTI_ITER_1_GPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
		}

		bufferedRenderProcess.setDoubleBuffers(bufferA, bufferB);
		if ((tests & TEST_SINGLE_ITER_CPU_ONLY) != NULL) {
			configuration.configureCPU(Renderer::ThreadConfiguration::ALL);
			configuration.configureEveryGPU(Renderer::ThreadConfiguration::NONE);
			Tests::runTest(testBufferedRenderProcess, "TEST_SINGLE_ITER_CPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
		}
		if (configuration.numDevices() > 0) {
			if ((tests & TEST_SINGLE_ITER_GPU_ONLY) != NULL) {
				configuration.configureCPU(Renderer::ThreadConfiguration::NONE);
				configuration.configureEveryGPU(2);
				Tests::runTest(testBufferedRenderProcess, "TEST_SINGLE_ITER_GPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
			}
			if ((tests & TEST_SINGLE_ITER_CPU_AND_GPU) != NULL) {
				configuration.configureCPU(Renderer::ThreadConfiguration::ALL);
				configuration.configureEveryGPU(2);
				Tests::runTest(testBufferedRenderProcess, "TEST_SINGLE_ITER_CPU_AND_GPU", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
			}
		}
		if ((tests & TEST_SINGLE_ITER_1_CPU_ONLY) != NULL) {
			configuration.configureCPU(Renderer::ThreadConfiguration::ONE);
			configuration.configureEveryGPU(Renderer::ThreadConfiguration::NONE);
			Tests::runTest(testBufferedRenderProcess, "TEST_SINGLE_ITER_1_CPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
		}
		if (((tests & TEST_SINGLE_ITER_1_GPU_ONLY) != NULL) && (configuration.numDevices() > 1)) {
			configuration.configureCPU(Renderer::ThreadConfiguration::NONE);
			configuration.configureEveryGPU(0);
			configuration.configureGPU(0, 2);
			Tests::runTest(testBufferedRenderProcess, "TEST_SINGLE_ITER_1_GPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
		}
	}
}
