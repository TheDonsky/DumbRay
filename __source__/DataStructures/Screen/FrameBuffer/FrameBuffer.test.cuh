#pragma once
#include"FrameBufferManager/FrameBufferManager.cuh"


namespace FrameBufferTest {
	enum RenderSettings {
		USE_GPU = 1,
		USE_CPU = 2
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
}
