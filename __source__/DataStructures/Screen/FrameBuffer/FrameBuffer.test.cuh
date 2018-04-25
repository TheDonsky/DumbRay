#pragma once
#include"FrameBuffer.cuh"
#include "../../../Namespaces/Tests/Tests.h"


namespace FrameBufferTest {
	namespace Private {
		void testPerformance(const std::string& bufferClassName, FrameBufferManager &front, FrameBufferManager &back);
	}

	template<typename Type, typename... Args>
	inline static void testPerformance(const std::string& bufferClassName, Args... args) {
		FrameBufferManager front, back;
		front.cpuHandle()->use<Type>(args...);
		back.cpuHandle()->use<Type>(args...);
		Private::testPerformance(bufferClassName, front, back);
	}
}
