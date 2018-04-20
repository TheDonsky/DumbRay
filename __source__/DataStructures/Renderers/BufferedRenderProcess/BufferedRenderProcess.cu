#include "BufferedRenderProcess.cuh"


#define FLAG_RENDER_PROCESS_THREAD_STARTED		0
#define FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED	1
#define FLAG_DOUBLE_BUFFERED_RENDER_MODE		2

#define FLAG(flag) (((uint16_t)1) << flag)
#define HAS_FLAG_ALL(all, flag)			((all & FLAG(flag)) != 0)
#define SET_FLAG_ALL(all, flag, value)	all = (value ? (all | FLAG(flag)) : (all & (~FLAG(flag))))
#define HAS_FLAG(flag) HAS_FLAG_ALL(flags, flag)
#define SET_FLAG(flag, value) SET_FLAG_ALL(flags, flag, value)


BufferedRenderProcess::BufferedRenderProcess() {
	flags = 0;
	targetRenderer = NULL;
	front = back = NULL;
	iterations = -1;
	bufferedWindow = NULL;
	targetWidth = targetHeight = -1;
}
BufferedRenderProcess::~BufferedRenderProcess() { end(); }


void BufferedRenderProcess::start() {
	std::lock_guard<std::mutex> guard(threadLock);
	if (HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED)) return;
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED, false);
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED, true);
	renderThread = std::thread(renderProcess, this);
}
void BufferedRenderProcess::end() {
	std::lock_guard<std::mutex> guard(threadLock);
	if (!HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED)) return;
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED, true);
	renderThread.join();
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED, false);
}

void BufferedRenderProcess::synchSettings() {
	std::unique_lock<std::mutex> lock(threadLock);
	if (!HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED)) return;
	synchCond.wait(lock);
}





void BufferedRenderProcess::renderProcess(BufferedRenderProcess *target) {
	while (true) {
		{
			std::lock_guard<std::mutex> guard(target->threadLock);
			if (HAS_FLAG_ALL(target->flags, FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED)) break;
		}
		BufferedRenderer *renderer;
		FrameBufferManager *frontBuffer, *backBuffer;
		int targetIterations;
		BufferedWindow *bufferedWindow;
		int targetWidth, targetHaight;
		{
			std::lock_guard<std::mutex> guard(target->settingsLock);
			renderer = target->targetRenderer;
			frontBuffer = target->front;
			backBuffer = target->back;
			targetIterations = target->iterations;
			targetWidth = target->targetWidth;
			targetHaight = target->targetHeight;
		}
		{
			// RENDER SINGLE ITERATION HERE...
		}
		target->synchCond.notify_all();
	}
}
