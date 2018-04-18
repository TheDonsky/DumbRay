#include "BufferedRenderProcess.cuh"


#define FLAG_RENDER_PROCESS_THREAD_STARTED		0
#define FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED	1
#define FLAG_DOUBLE_BUFFERED_RENDER_MODE		2

#define FLAG(flag) (((uint16_t)1) << flag)
#define HAS_FLAG(flag)			((flags & FLAG(flag)) != 0)
#define SET_FLAG(flag, value)	flags = (value ? (flags | FLAG(flag)) : (flags & (~FLAG(flag))))


BufferedRenderProcess::BufferedRenderProcess() {
	flags = 0;
}
BufferedRenderProcess::~BufferedRenderProcess() { end(); }


void BufferedRenderProcess::start() {
	std::lock_guard<std::mutex> guard(settingsLock);
	if (HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED)) return;
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED, false);
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED, true);
	renderThread = std::thread(renderProcess, this);
}
void BufferedRenderProcess::end() {
	std::lock_guard<std::mutex> guard(settingsLock);
	if (!HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED)) return;
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED, true);
	renderThread.join();
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED, false);
}

void BufferedRenderProcess::synchSettings() {
	std::unique_lock<std::mutex> lock(settingsLock);
	synchCond.wait(lock);
}





void BufferedRenderProcess::renderProcess(BufferedRenderProcess *target) {
	target->renderProcessThread();
}
void BufferedRenderProcess::renderProcessThread() {
	while(!HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED)) {
		{
			std::lock_guard<std::mutex> guard(settingsLock);
			// EXTRACT CURRENT SETTING HERE...
		}
		{
			// RENDER SINGLE ITERATION HERE...
		}
		synchCond.notify_all();
	}
}
