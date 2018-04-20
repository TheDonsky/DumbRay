#include "BufferedRenderer.cuh"



BufferedRenderer::BufferedRenderer(const ThreadConfiguration &configuration, FrameBufferManager *buffer) : Renderer(configuration) { setFrameBuffer(buffer); }
BufferedRenderer::BufferedRenderer(ThreadConfiguration &&configuration, FrameBufferManager *buffer) : Renderer(configuration) { setFrameBuffer(buffer); }
void BufferedRenderer::setFrameBuffer(FrameBufferManager *buffer) {  if (manager != buffer) { manager = buffer; resetIterations(); } }
FrameBufferManager *BufferedRenderer::getFrameBuffer() { return manager; }
const FrameBufferManager *BufferedRenderer::getFrameBuffer()const { return manager; }



