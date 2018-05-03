#pragma once
#include "../FrameBuffer/FrameBuffer.cuh"
#include "../../../Namespaces/Windows/Windows.h"
#include <thread>
#include <atomic>
#include <mutex>


class BufferedWindow {
public:
	enum Options {
		SYNCH_FRAME_BUFFER_FROM_DEVICE = 1
	};
	typedef uint16_t OptionFlags;

	BufferedWindow(
		OptionFlags optionFlags = 0,
		const char *windowName = "BufferedWindow",
		FrameBufferManager *frameBufferManager = NULL,
		int renderingDeviceId = 0);
	~BufferedWindow();

	void setBuffer(FrameBufferManager *frameBufferManager);
	bool trySetBuffer(FrameBufferManager *frameBufferManager);
	void notifyChange();

	bool getWindowResolution(int &width, int &height)const;
	bool windowClosed()const;
	void closeWindow();

	size_t framesDisplayed()const;


private:
	enum State {
		BUFFERED_WINDOW_THREAD_FINISHED = 1,
		BUFFERED_WINDOW_SHOULD_EXIT = 2,
		WINDOW_DESTROYED = 4
	};
	typedef uint16_t StateFlags;

	BufferedWindow(const BufferedWindow &other) {};
	BufferedWindow& operator=(const BufferedWindow &other) { return (*this); };

	OptionFlags options;
	int deviceId;
	volatile StateFlags state;

	char windowMemory[sizeof(Windows::Window)];
	std::mutex bufferLock;
	std::condition_variable bufferCond;
	
	volatile FrameBufferManager *bufferManager;
	std::atomic<size_t> numFramesDisplayed;

	std::thread windowThread;

	Windows::Window &window();
	const Windows::Window &window()const;

	static void bufferedWindowThread(BufferedWindow *bufferedWindow);
};
