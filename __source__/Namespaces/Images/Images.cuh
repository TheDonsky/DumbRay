#pragma once
#include"../../DataStructures/Screen/FrameBuffer/FrameBuffer.cuh"
#include"../../DataStructures/Objects/Components/Texture/Texture.cuh"
#include<string>


namespace Images {
	enum Error {
		IMAGES_NO_ERROR = 0,
		IMAGES_ERROR_FRAME_BUFFER_INVALID = 1,
		IMAGES_ERROR_ALLOCATION_FAILED = 2,
		IMAGES_ERROR_EXTERNAL_LIBRARY_FAILED = 3
	};

	Error saveBufferPNG(const FrameBuffer &buffer, const std::string &filename);

	Error getTexturePNG(Texture &texture, const std::string &filename);
}

