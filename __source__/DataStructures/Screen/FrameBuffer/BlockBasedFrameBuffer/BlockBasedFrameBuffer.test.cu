#include "BlockBasedFrameBuffer.cuh"
#include "../FrameBuffer.test.cuh"

namespace BlockBasedFrameBufferTest {
	void test() { FrameBufferTest::fullPerformanceTest<BlockBuffer>("BlockBuffer"); }
}
