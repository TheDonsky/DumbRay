#include "BlockBasedFrameBuffer.cuh"

std::mutex BlockBufferData::deviceReferenceLock;
BlockBufferData::DeviceReferenceMirrors BlockBufferData::deviceReferenceMirrors;
