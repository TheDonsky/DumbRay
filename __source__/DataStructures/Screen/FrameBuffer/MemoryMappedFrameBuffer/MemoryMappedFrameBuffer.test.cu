#include "MemoryMappedFrameBuffer.cuh"
#include "../FrameBuffer.test.cuh"

namespace MemoryMappedFrameBufferTest {
	void test() { FrameBufferTest::testPerformance<MemoryMappedFrameBuffer>("MemoryMappedFrameBuffer"); }
}
