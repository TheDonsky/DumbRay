#include "BufferedRenderProcess.test.cuh"
#include "../../../Namespaces/Tests/Tests.h"
#include "../../../Namespaces/Images/Images.cuh"
#include <string>
#include <cctype>

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
			bool shouldSynchFromDevice = (!((configuration.numHostThreads() > 0) || (configuration.numActiveDevices() > 1)));
			BufferedWindow bufferedWindow(shouldSynchFromDevice ? BufferedWindow::SYNCH_FRAME_BUFFER_FROM_DEVICE : 0);
			process->setRenderer(renderer);
			process->setTargetDisplayWindow(&bufferedWindow);
			FramerateLogger logger;
			logger.setBufferedWindow(&bufferedWindow);
			logger.registerIterationCompletionCallback(process);
			logger.start();
			process->start();
			while (!bufferedWindow.windowClosed()) std::this_thread::sleep_for(std::chrono::milliseconds(32));
			process->end();
			if ((renderer->getFrameBuffer() != NULL) && (renderer->getFrameBuffer()->cpuHandle() != NULL)) {
				std::cout << "Enter name ending with \".png\" to save the render: ";
				std::string rawLine;
				std::getline(std::cin, rawLine);
				int start = 0; while ((start < rawLine.length()) && std::isspace(rawLine[start])) start++;
				int end = (int)rawLine.size(); while ((end > 0) && std::isspace(rawLine[end - 1])) end--;
				std::string line; for (int i = start; i < end; i++) line += rawLine[i];
				if ((line.length() > 4) && line.substr(line.length() - 4) == ".png") {
					std::cout << "Saving to \"" << line << "\"..." << std::endl;
					bool failed = false;
					if (shouldSynchFromDevice) {
						if (cudaSetDevice(0) != cudaSuccess) failed = true;
						else if (renderer->getFrameBuffer()->gpuHandle(0) == NULL) failed = true;
						else if (!renderer->getFrameBuffer()->cpuHandle()->updateHostBlocks(
							renderer->getFrameBuffer()->gpuHandle(0), 0, renderer->getFrameBuffer()->cpuHandle()->getBlockCount())) failed = true;
					}
					if (!failed) {
						Images::Error error = Images::saveBufferPNG(*renderer->getFrameBuffer()->cpuHandle(), line);
						if (error != Images::IMAGES_NO_ERROR)
							std::cout << "Failed to save image. Error code: " << error << std::endl;
						else std::cout << "Image saved successfuly;" << std::endl;
					}
					else std::cout << "Failed to load image from device..." << std::endl;
				}
			}
			delete renderer;
		}
	}

#define NUM_THREADS_PER_GPU 2
	
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
				configuration.configureEveryGPU(NUM_THREADS_PER_GPU);
				Tests::runTest(testBufferedRenderProcess, "TEST_MULTI_ITER_GPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
			}
			if ((tests & TEST_MULTI_ITER_CPU_AND_GPU) != NULL) {
				configuration.configureCPU(Renderer::ThreadConfiguration::ALL_BUT_GPU_THREADS);
				configuration.configureEveryGPU(NUM_THREADS_PER_GPU);
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
			configuration.configureGPU(0, NUM_THREADS_PER_GPU);
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
				configuration.configureEveryGPU(NUM_THREADS_PER_GPU);
				Tests::runTest(testBufferedRenderProcess, "TEST_SINGLE_ITER_GPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
			}
			if ((tests & TEST_SINGLE_ITER_CPU_AND_GPU) != NULL) {
				configuration.configureCPU(Renderer::ThreadConfiguration::ALL_BUT_GPU_THREADS);
				configuration.configureEveryGPU(NUM_THREADS_PER_GPU);
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
			configuration.configureGPU(0, NUM_THREADS_PER_GPU);
			Tests::runTest(testBufferedRenderProcess, "TEST_SINGLE_ITER_1_GPU_ONLY", bufferedRendererCreateFunction, createFnAux, configuration, &bufferedRenderProcess);
		}
	}
}
