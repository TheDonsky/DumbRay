#include"FrameBuffer.test.cuh"
#include"../../Renderers/Renderer/Renderer.cuh"
#include "../../../Namespaces/Windows/Windows.h"


namespace FrameBufferTest {
	namespace {
		class TestRender : public Renderer {
		private:
			FrameBufferManager *front, *back;


		public:
			void useBuffers(FrameBufferManager &frontBuffer, FrameBufferManager &backBuffer) {
				front = &frontBuffer;
				back = &backBuffer;
			}

			void test() {
				Windows::Window window;
				while (!window.dead()) {

				}
			}
		};
	}
	
	namespace Private {
		void testPerformance(FrameBufferManager &front, FrameBufferManager &back) {

		}
	}
}

