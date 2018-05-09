#include"Images.cuh"
#include"lodepng.h"
#include<vector>

namespace Images {
	namespace {
		inline static void translateChannelFloatToByte(const float channel, unsigned char &byte) {
			float value = (255.0f * channel);
			if (value > 255.0f) byte = 255;
			else if (value < 0.0f) byte = 0;
			else byte = ((unsigned char)value);
		}

		inline static void translateToBytes(const Color &color, unsigned char &r, unsigned char &g, unsigned char &b, unsigned char &a) {
			translateChannelFloatToByte(color.r, r);
			translateChannelFloatToByte(color.g, g);
			translateChannelFloatToByte(color.b, b);
			translateChannelFloatToByte(color.a, a);
		}
	}

	Error saveBufferPNG(const FrameBuffer &buffer, const std::string &filename) {
		int width, height;
		buffer.getSize(&width, &height);
		if (width < 0 || height < 0) return IMAGES_ERROR_FRAME_BUFFER_INVALID;
		std::vector<unsigned char> bufferBytes;
		bufferBytes.resize(width * height * 4, 0);
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				int pos = (4 * ((width * y) + x));
				translateToBytes(buffer.getColor(x, y), bufferBytes[pos], bufferBytes[pos + 1], bufferBytes[pos + 2], bufferBytes[pos + 3]);
			}
		unsigned int lodePngError = lodepng::encode(filename, bufferBytes, width, height);
		if (lodePngError) return IMAGES_ERROR_EXTERNAL_LIBRARY_FAILED;
		return IMAGES_NO_ERROR;
	}
}
