#include "MemoryMappedFrameBuffer.cuh"
#include "../FrameBuffer.test.cuh"

namespace MemoryMappedFrameBufferTest {
	void test() { FrameBufferTest::fullPerformanceTest<MemoryMappedFrameBuffer>("MemoryMappedFrameBuffer"); }
}
