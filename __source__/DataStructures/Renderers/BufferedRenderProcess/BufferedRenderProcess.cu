#include "BufferedRenderProcess.cuh"
#include <chrono>
#include <thread>


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

	iterationCompletionCallback = NULL;
	iterationCompletionCallbackArg = NULL;

	renderCompletionCallback = NULL;
	renderCompletionCallbackArg = NULL;

	alreadyRenderedCallback = NULL;
	alreadyRenderedCallbackArg = NULL;

	errorOnIteration = NULL;
	errorOnIterationArg = NULL;

	renderClocks = 0;
}
BufferedRenderProcess::~BufferedRenderProcess() { end(); }


void BufferedRenderProcess::start() {
	std::lock_guard<std::mutex> guard(threadLock);
	if (HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED)) return;
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED, false);
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED, true);
	renderThread = std::thread(renderProcess, this);
	startClock = clock();
}
void BufferedRenderProcess::end() {
	std::lock_guard<std::mutex> guard(threadLock);
	if (!HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED)) return;
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED, true);
	renderThread.join();
	renderClocks = renderTime();
	SET_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED, false);
}

void BufferedRenderProcess::lockSettings() { settingsLock.lock(); }
void BufferedRenderProcess::unlockSettings() { settingsLock.unlock(); }

void BufferedRenderProcess::setRenderer(BufferedRenderer *renderer, bool lock) {
	if (lock) lockSettings(); targetRenderer = renderer; if (lock) unlockSettings();
}
void BufferedRenderProcess::setBuffer(FrameBufferManager *buffer, bool lock) {
	if (lock) lockSettings(); front = back = buffer; if (lock) unlockSettings();
}
void BufferedRenderProcess::setDoubleBuffers(FrameBufferManager *a, FrameBufferManager *b, bool lock) {
	if (lock) lockSettings(); front = a; back = b; if (lock) unlockSettings();
}
void BufferedRenderProcess::setTargetIterations(int targetIterations, bool lock) {
	if (lock) lockSettings(); iterations = targetIterations; if (lock) unlockSettings();
}
void BufferedRenderProcess::setInfinateTargetIterations(bool lock) { setTargetIterations(-1, lock); }
void BufferedRenderProcess::setTargetDisplayWindow(BufferedWindow *window, bool lock) {
	if (lock) lockSettings(); bufferedWindow = window; if (lock) unlockSettings();
}
void BufferedRenderProcess::setTargetResolution(int width, int height, bool lock) {
	if (lock) lockSettings(); targetWidth = width; targetHeight = height; if (lock) unlockSettings();
}
void BufferedRenderProcess::setTargetResolutionToWindowSize(bool lock) { setTargetResolution(-1, -1, lock); }

void BufferedRenderProcess::setIterationCompletionCallback(Callback callback, void *arg, bool lock) {
	if (lock) lockSettings(); iterationCompletionCallback = callback; iterationCompletionCallbackArg = arg; if (lock) unlockSettings();
}
void BufferedRenderProcess::setRenderCompletionCallback(Callback callback, void *arg, bool lock) {
	if (lock) lockSettings(); renderCompletionCallback = callback; renderCompletionCallbackArg = arg; if (lock) unlockSettings();
}
void BufferedRenderProcess::setAlreadyRenderedCallback(Callback callback, void *arg, bool lock) {
	if (lock) lockSettings(); alreadyRenderedCallback = callback; alreadyRenderedCallbackArg = arg; if (lock) unlockSettings();
}
void BufferedRenderProcess::setErrorOnIteration(Callback callback, void *arg, bool lock) {
	if (lock) lockSettings(); errorOnIteration = callback; errorOnIterationArg = arg; if (lock) unlockSettings();
}

void BufferedRenderProcess::synchSettings(bool alreadyLocked) {
	if (alreadyLocked) unlockSettings();
	std::unique_lock<std::mutex> lock(settingsLock);
	if (!HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED)) return;
	synchCond.wait(lock);
}

long long BufferedRenderProcess::renderTime()const {
	return (renderClocks + (HAS_FLAG(FLAG_RENDER_PROCESS_THREAD_STARTED) ? (clock() - startClock) : 0));
}





void BufferedRenderProcess::renderProcess(BufferedRenderProcess *target) {
	bool swapBuffers = false;
	while (true) {
		/* ________________________ */
		/* __EXIT_IF_KILL_ISSUED__: */
		if (HAS_FLAG_ALL(target->flags, FLAG_RENDER_PROCESS_THREAD_KILL_ISSUED)) break;

		/* ______________________ */
		/* __SETTING_VARIABLES__: */
		BufferedRenderer *renderer;
		FrameBufferManager *frontBuffer, *backBuffer;
		int targetIterations;
		BufferedWindow *bufferedWindow;
		int targetWidth, targetHeight;

		Callback iterationCompletionCallback;
		void* iterationCompletionCallbackArg;
		Callback renderCompletionCallback;
		void* renderCompletionCallbackArg;
		Callback alreadyRenderedCallback;
		void* alreadyRenderedCallbackArg;
		Callback errorOnIteration;
		void* errorOnIterationArg;

		/* ______________________ */
		/* __SETTING_EXECUTION__: */
		{
			std::lock_guard<std::mutex> guard(target->settingsLock);
			
			renderer = target->targetRenderer;
			frontBuffer = (swapBuffers ? target->back : target->front);
			backBuffer = (swapBuffers ? target->front : target->back);	
			targetIterations = target->iterations;
			bufferedWindow = target->bufferedWindow;
			targetWidth = target->targetWidth;
			targetHeight = target->targetHeight;

			iterationCompletionCallback = target->iterationCompletionCallback;
			iterationCompletionCallbackArg = target->iterationCompletionCallbackArg;
			renderCompletionCallback = target->renderCompletionCallback;
			renderCompletionCallbackArg = target->renderCompletionCallbackArg;
			alreadyRenderedCallback = target->alreadyRenderedCallback;
			alreadyRenderedCallbackArg = target->alreadyRenderedCallbackArg;

			errorOnIteration = target->errorOnIteration;
			errorOnIterationArg = target->errorOnIterationArg;

			target->synchCond.notify_all();
		}

		/* __________________________________ */
		/* __BACK_BUFFER_RESOLUTION_CHANGE__: */
		if (backBuffer != NULL && backBuffer->cpuHandle() != NULL) {
			if ((targetWidth < 0) || (targetHeight < 0) && (bufferedWindow != NULL)) {
				if (!bufferedWindow->getWindowResolution(targetWidth, targetHeight)) targetWidth = targetHeight = 32;
			}
			else {
				targetWidth = max(abs(targetWidth), 1);
				targetHeight = max(abs(targetHeight), 1);
			}
			int imageWidth, imageHeight;
			backBuffer->cpuHandle()->getSize(&imageWidth, &imageHeight);
			if ((imageWidth != targetWidth) || (imageHeight != targetHeight)) {
				bool bufferedWindowSharesBuffer = ((backBuffer == frontBuffer) && (bufferedWindow != NULL));
				if (bufferedWindowSharesBuffer) bufferedWindow->setBuffer(NULL);
				backBuffer->cpuHandle()->setResolution(targetWidth, targetHeight);
				backBuffer->makeDirty();
				if (bufferedWindowSharesBuffer) bufferedWindow->setBuffer(frontBuffer);
				if (renderer != NULL) {
					renderer->resetIterations();
					target->renderClocks = 0;
					target->startClock = clock();
				}
			}
		}

		/* _____________________ */
		/* __RENDER_ITERATION__: */
		{
			bool bufferedWindowHasToBeInformed = true;
			if (renderer != NULL) {
				renderer->setFrameBuffer(backBuffer);
				if (renderer->iteration() != targetIterations) {
					if (!renderer->iterate()) { 
						if (errorOnIteration != NULL) errorOnIteration(errorOnIterationArg);
						continue; 
					}
					if (backBuffer != frontBuffer) {
						bool shouldSwap = true;
						if (bufferedWindow != NULL) {
							bufferedWindowHasToBeInformed = false;
							if (bufferedWindow->trySetBuffer(backBuffer)) bufferedWindow->notifyChange();
							else shouldSwap = false;
						}
						if (shouldSwap) swapBuffers = (!swapBuffers);
						renderer->resetIterations();
						target->renderClocks = 0;
						target->startClock = clock();
					}
					if (iterationCompletionCallback != NULL) iterationCompletionCallback(iterationCompletionCallbackArg);
					if ((renderer->iteration() == targetIterations) && (renderCompletionCallback != NULL)) renderCompletionCallback(renderCompletionCallbackArg);
				}
				else if (alreadyRenderedCallback != NULL) alreadyRenderedCallback(alreadyRenderedCallbackArg);
			}
			else std::this_thread::sleep_for(std::chrono::milliseconds(8));
			if (bufferedWindowHasToBeInformed && (bufferedWindow != NULL)) {
				bufferedWindow->trySetBuffer(backBuffer);
				bufferedWindow->notifyChange();
			}
		}
	}
	target->synchCond.notify_all();
}
