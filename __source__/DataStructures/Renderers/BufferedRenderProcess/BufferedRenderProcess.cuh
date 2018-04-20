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

	void setRenderer(BufferedRenderer *renderer);
	void setBuffer(FrameBufferManager *buffer);
	void setDoubleBuffers(FrameBufferManager *a, FrameBufferManager *b);
	void setTargetIterations(int targetIterations);
	void setInfinateTargetIterations();
	void setTargetDisplayWindows(BufferedWindow *window);
	void setTargetResolution(int width, int height);
	void setTargetResolutionToWindowSize();

	void synchSettings();


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

	static void renderProcess(BufferedRenderProcess *target);
};

