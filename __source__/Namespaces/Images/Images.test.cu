#include "Images.test.cuh"
#include "Images.cuh"
#include "../../DataStructures/Screen/FrameBuffer/BlockBasedFrameBuffer/BlockBasedFrameBuffer.cuh"
#include "../Tests/Tests.h"
#include "../Windows/Windows.h"
#include <thread>
#include <chrono>


namespace ImagesTest {
	namespace {
		static void savePngTestCase() {
			FrameBuffer buffer;
			buffer.use<BlockBuffer>();
			const int width = 512, height = 256;
			if (!buffer.setResolution(width, height)) {
				std::cout << "Fail: Could not set buffer resolution..." << std::endl;
				return;
			}
			std::cout << "Generating image..." << std::endl;
			for (int x = 0; x < width; x++)
				for (int y = 0; y < height; y++)
					buffer.setColor(x, y, Color(((float)(255 * !(x & y))) / 255.0f, ((float)(x ^ y)) / 255.0f, ((float)(x | y)) / 255.0f, 1.0f));
			std::cout << "Saving image..." << std::endl;
			Images::Error error = Images::saveBufferPNG(buffer, "ImagesTest_testSavePng.png");
			std::cout << "Status: " << error << std::endl;
			std::cout << "Should have saved whatever's on screen:";
			
			Windows::Window window("Test window");
			window.updateFromHost(buffer);
			int w = -1, h = -1;
			while (!window.dead()) {
				int newW, newH;
				if (!window.getDimensions(newW, newH)) break;
				if (w != newW || h != newH) {
					w = newW;
					h = newH;
					window.updateFromHost(buffer);
				}
				std::this_thread::sleep_for(std::chrono::microseconds(32));
			}
			std::cout << std::endl;
		}

		static void loadPngTestCase() {
			Texture texture;
			std::cout << "Loading image..." << std::endl;
			Images::Error error = Images::getTexturePNG(texture, "ImagesTest_testSavePng.png");
			std::cout << "Status: " << error << std::endl;
			Windows::Window window("Test window");
			window.updateFrameHost(texture[0], texture.width(), texture.height());
			int w = -1, h = -1;
			while (!window.dead()) {
				int newW, newH;
				if (!window.getDimensions(newW, newH)) break;
				if (w != newW || h != newH) {
					w = newW;
					h = newH;
					window.updateFrameHost(texture[0], texture.width(), texture.height());
				}
				std::this_thread::sleep_for(std::chrono::microseconds(32));
			}
			std::cout << std::endl;
		}
	}
	
	void testSavePng() {
		Tests::runTest(savePngTestCase, "Testing Images::savePNG");
	}

	void testLoadPng() {
		Tests::runTest(loadPngTestCase, "Testing Images::loadPNG");
	}
}
