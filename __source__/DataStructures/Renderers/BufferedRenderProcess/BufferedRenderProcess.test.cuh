#pragma once
#include "BufferedRenderProcess.cuh"
#include"../../Screen/BufferedWindow/BufferedWindow.cuh"


namespace BufferedRenderProcessTest {
	enum TestsCases {
		TEST_MULTI_ITER_CPU_ONLY = 1,
		TEST_MULTI_ITER_GPU_ONLY = 2,
		TEST_MULTI_ITER_CPU_AND_GPU = 4,
		TEST_MULTI_ITER_1_CPU_ONLY = 8,
		TEST_MULTI_ITER_1_GPU_ONLY = 16,
		TEST_SINGLE_ITER_CPU_ONLY = 32,
		TEST_SINGLE_ITER_GPU_ONLY = 64,
		TEST_SINGLE_ITER_CPU_AND_GPU = 128,
		TEST_SINGLE_ITER_1_CPU_ONLY = 256,
		TEST_SINGLE_ITER_1_GPU_ONLY = 512,
		TEST_RUN_FULL_GAUNTLET = 1023
	};

	class FramerateLogger {
	public:
		void start();
		void setBufferedWindow(BufferedWindow *bufferedWindow);
		void registerIterationCompletionCallback(BufferedRenderProcess *process);

	private:
		volatile uint32_t renderedFrames;
		volatile uint32_t lastRenderedFrames, lastDisplayedFrames;
		volatile clock_t lastTime;
		BufferedWindow *displayWindow;

		static void iterationCompletionCallback(void *testCase);
	};

	void runTestGauntlet(
		BufferedRenderer*(*bufferedRendererCreateFunction)(const Renderer::ThreadConfiguration &configuration),
		FrameBufferManager *bufferA, FrameBufferManager *bufferB, uint32_t tests = TEST_RUN_FULL_GAUNTLET);

	template<typename BufferedRendererType>
	inline BufferedRenderer *makeBufferedRenderer(const Renderer::ThreadConfiguration &configuration) {
		return new BufferedRendererType(configuration);
	}

	template<typename BufferedRendererType>
	inline void runTestGauntlet(FrameBufferManager *bufferA, FrameBufferManager *bufferB, uint32_t tests = TEST_RUN_FULL_GAUNTLET) {
		runTestGauntlet(makeBufferedRenderer<BufferedRendererType>, bufferA, bufferB, tests);
	}
}

