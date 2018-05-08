#pragma once
#include"../../DataStructures/Screen/FrameBuffer/FrameBuffer.cuh"
#include<string>


namespace Images {
	enum Error {
		NO_ERROR = 0,
		ERROR_FRAME_BUFFER_INVALID = 1,
		ERROR_ALLOCATION_FAILED = 2,
		ERROR_EXTERNAL_LIBRARY_FAILED = 3
	};

	Error saveBufferPNG(const FrameBuffer &buffer, const std::string &filename);
}

