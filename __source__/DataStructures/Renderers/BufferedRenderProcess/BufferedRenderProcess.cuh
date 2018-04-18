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

	void synchSettings();


private:
	volatile uint16_t flags;
	std::mutex settingsLock;
	std::thread renderThread;
	std::condition_variable synchCond;

	static void renderProcess(BufferedRenderProcess *target);
	void renderProcessThread();
};

