#pragma once
#include "../BufferedRenderer/BufferedRenderer.cuh"
#include "../../Screen/BufferedWindow/BufferedWindow.cuh"
#include <mutex>
#include <thread>





class BufferedRenderProcess {
public:
	BufferedRenderProcess();
	~BufferedRenderProcess();

	void start();
	void end();

	void lockSettings();
	void unlockSettings();

	void setRenderer(BufferedRenderer *renderer, bool lock = false);
	void setBuffer(FrameBufferManager *buffer, bool lock = false);
	void setDoubleBuffers(FrameBufferManager *a, FrameBufferManager *b, bool lock = false);
	void setTargetIterations(int targetIterations, bool lock = false);
	void setInfinateTargetIterations(bool lock = false);
	void setTargetDisplayWindow(BufferedWindow *window, bool lock = false);
	void setTargetResolution(int width, int height, bool lock = false);
	void setTargetResolutionToWindowSize(bool lock = false);

	typedef void(*Callback)(void *arg);
	void setIterationCompletionCallback(Callback callback, void *arg, bool lock = false);
	void setRenderCompletionCallback(Callback callback, void *arg, bool lock = false);
	void setAlreadyRenderedCallback(Callback callback, void *arg, bool lock = false);

	void setErrorOnResolutionChange(Callback callback, void *arg, bool lock = false);
	void setErrorOnIteration(Callback callback, void *arg, bool lock = false);

	void synchSettings(bool alreadyLocked = false);


private:
	volatile uint16_t flags;
	std::mutex threadLock;
	std::mutex settingsLock;
	std::thread renderThread;
	std::condition_variable synchCond;

	BufferedRenderer *targetRenderer;
	FrameBufferManager *front, *back;
	int iterations;
	BufferedWindow *bufferedWindow;
	int targetWidth, targetHeight;

	Callback iterationCompletionCallback;
	void* iterationCompletionCallbackArg;
	Callback renderCompletionCallback;
	void* renderCompletionCallbackArg;
	Callback alreadyRenderedCallback;
	void* alreadyRenderedCallbackArg;

	Callback errorOnResolutionChange;
	void* errorOnResolutionChangeArg;
	Callback errorOnIteration;
	void* errorOnIterationArg;

	static void renderProcess(BufferedRenderProcess *target);
};

