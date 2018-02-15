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

	template<typename Type>
	inline static void testPerformance(Flags settings = (USE_GPU | USE_CPU)) {
		FrameBufferManager front, back;
		front.cpuHandle()->use<Type>();
		back.cpuHandle()->use<Type>();
		Private::testPerformance(front, back, settings);
	}
}
