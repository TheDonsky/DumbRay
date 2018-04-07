#pragma once
#include"FrameBufferManager/FrameBufferManager.cuh"
#include "../../../Namespaces/Tests/Tests.h"


namespace FrameBufferTest {
	enum RenderSettings {
		USE_GPU = 1,
		USE_CPU = 2,
		UPDATE_SCREEN_FROM_DEVICE = 4,
		TEST_FOR_SINGLE_ITERATION = 8,
		TEST_SINGLE_GPU_ONLY = 16
	};
	typedef unsigned int Flags;
	
	namespace Private {
		void testPerformance(FrameBufferManager &front, FrameBufferManager &back, Flags flags);
	}

	template<typename Type, typename... Args>
	inline static void testPerformance(Flags settings = (USE_GPU | USE_CPU), Args... args) {
		FrameBufferManager front, back;
		front.cpuHandle()->use<Type>(args...);
		back.cpuHandle()->use<Type>(args...);
		Private::testPerformance(front, back, settings);
	}

	template<typename Type, typename... Args>
	inline static void fullPerformanceTest(const std::string &typeName) {
		Tests::runTest(
			testPerformance<Type>,
			"Testing " + typeName + " (GPU & CPU; turn off the window to quit)",
			USE_CPU | USE_GPU);

		Tests::runTest(
			testPerformance<Type>,
			"Testing " + typeName + " (CPU; turn off the window to quit)",
			USE_CPU);

		Tests::runTest(
			testPerformance<Type>,
			"Testing " + typeName + " (GPU; turn off the window to quit)",
			USE_GPU);

		Tests::runTest(
			testPerformance<Type>,
			"Testing " + typeName + " (GPU & CPU single iteration)",
			USE_CPU | USE_GPU | TEST_FOR_SINGLE_ITERATION);

		Tests::runTest(
			testPerformance<Type>,
			"Testing " + typeName + " (CPU single iteration)",
			USE_CPU | TEST_FOR_SINGLE_ITERATION);

		Tests::runTest(
			testPerformance<Type>,
			"Testing " + typeName + " (GPU single iteration)",
			USE_GPU | TEST_FOR_SINGLE_ITERATION);

		Tests::runTest(
			testPerformance<Type>,
			"Testing " + typeName + " (single GPU; single iteration)",
			USE_GPU | TEST_FOR_SINGLE_ITERATION | TEST_SINGLE_GPU_ONLY);
	}
}
