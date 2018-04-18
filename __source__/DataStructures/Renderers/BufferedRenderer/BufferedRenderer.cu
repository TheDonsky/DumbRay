#include "BufferedRenderer.cuh"



BufferedRenderer::BufferedRenderer(const ThreadConfiguration &configuration, FrameBufferManager *buffer = NULL) : Renderer(configuration) { setFrameBuffer(buffer); }
BufferedRenderer::BufferedRenderer(ThreadConfiguration &&configuration, FrameBufferManager *buffer = NULL) : Renderer(configuration) { setFrameBuffer(buffer); }
void BufferedRenderer::setFrameBuffer(FrameBufferManager *buffer) { manager = buffer; }
FrameBufferManager *BufferedRenderer::getFrameBuffer() { return manager; }
const FrameBufferManager *BufferedRenderer::getFrameBuffer()const { return manager; }



