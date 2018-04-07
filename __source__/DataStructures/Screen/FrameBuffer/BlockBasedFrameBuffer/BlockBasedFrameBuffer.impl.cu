#include "BlockBasedFrameBuffer.cuh"

std::mutex BlockBasedFrameBuffer::deviceReferenceLock;
BlockBasedFrameBuffer::DeviceReferenceMirrors BlockBasedFrameBuffer::deviceReferenceMirrors;
