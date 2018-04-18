#pragma once
#include "../BufferedRenderer/BufferedRenderer.cuh"
#include "../../Screen/BufferedWindow/BufferedWindow.cuh"





class BufferedRenderProcess {
public:
	BufferedRenderProcess();
	virtual ~BufferedRenderProcess();
	
	enum Status {
		STATUS_OK = 0,
		STATUS_ERROR_PROCESS_ALREADY_STARTED = 1,
		STATUS_ERROR_FAILED_TO_START_PROCESS = 2,
		STATUS_ERROR_PROCESS_HAS_NOT_STARTED = 3
	};

	Status setRenderer(BufferedRenderer *renderer);
	Status setFrontBuffer(FrameBuffer *frontBuffer);
	Status setBackBuffer(FrameBuffer *backBuffer);
	Status setDisplayWindow(BufferedWindow *window);
	Status setFixedTargetResolution(uint32_t width, uint32_t height);
	Status linkTargetResolutionToWindowSize();

	Status startRenderProcess();
	Status stopRenderProcess();


private:

};

