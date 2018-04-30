#pragma once
#include "../Renderer/Renderer.cuh"
#include "../../Screen/FrameBuffer/FrameBuffer.cuh"





class BufferedRenderer : public Renderer {
public:
	BufferedRenderer(const ThreadConfiguration &configuration, FrameBufferManager *buffer = NULL);
	BufferedRenderer(ThreadConfiguration &&configuration, FrameBufferManager *manager = NULL);
	void setFrameBuffer(FrameBufferManager *buffer);
	FrameBufferManager *getFrameBuffer();
	const FrameBufferManager *getFrameBuffer()const;





private:
	volatile FrameBufferManager volatile *manager;
};
