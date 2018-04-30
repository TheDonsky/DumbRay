#include "BufferedWindow.cuh"
#include <new>


BufferedWindow::BufferedWindow(OptionFlags optionFlags, const char *windowName, FrameBufferManager *frameBufferManager, int renderingDeviceId) {
	options = optionFlags;
	bufferManager = frameBufferManager;
	deviceId = renderingDeviceId;
	state = 0;
	numFramesDisplayed = 0;
	new(&window()) Windows::Window(windowName);
	windowThread = std::thread(bufferedWindowThread, this);
}
BufferedWindow::~BufferedWindow() { closeWindow(); }

void BufferedWindow::setBuffer(FrameBufferManager *frameBufferManager) {
	std::lock_guard<std::mutex> guard(bufferLock);
	bufferManager = frameBufferManager;
}
bool BufferedWindow::trySetBuffer(FrameBufferManager *frameBufferManager) {
	if (bufferLock.try_lock()) {
		bufferManager = frameBufferManager;
		bufferLock.unlock();
		return true;
	}
	return false;
}
void BufferedWindow::notifyChange() { bufferCond.notify_one();  }

bool BufferedWindow::getWindowResolution(int &width, int &height)const { return window().getDimensions(width, height); }
bool BufferedWindow::windowClosed()const { return window().dead(); }
void BufferedWindow::closeWindow() {
	while (true) {
		std::lock_guard<std::mutex> guard(bufferLock);
		if ((state & BUFFERED_WINDOW_THREAD_FINISHED) != 0) break;
		state |= BUFFERED_WINDOW_SHOULD_EXIT;
		bufferCond.notify_one();
	}
	std::lock_guard<std::mutex> guard(bufferLock);
	if ((state & WINDOW_DESTROYED) != 0) return;
	state |= WINDOW_DESTROYED;
	windowThread.join();
	window().~Window();
}

size_t BufferedWindow::framesDisplayed()const { return numFramesDisplayed; }

Windows::Window &BufferedWindow::window() { return (*((Windows::Window*)windowMemory)); }
const Windows::Window &BufferedWindow::window()const { return (*((const Windows::Window*)windowMemory)); }

void BufferedWindow::bufferedWindowThread(BufferedWindow *bufferedWindow) {
	bool deviceSynchNeeded = ((bufferedWindow->options & SYNCH_FRAME_BUFFER_FROM_DEVICE) != 0);
	if (deviceSynchNeeded) if (cudaSetDevice(bufferedWindow->deviceId != cudaSuccess)) {
		bufferedWindow->state |= BUFFERED_WINDOW_THREAD_FINISHED;
		return;
	}
	while (true) {
		std::unique_lock<std::mutex> uniqueLock(bufferedWindow->bufferLock);
		bufferedWindow->bufferCond.wait(uniqueLock);
		if (bufferedWindow->windowClosed() || ((bufferedWindow->state & BUFFERED_WINDOW_SHOULD_EXIT) != 0)) {
			bufferedWindow->state |= BUFFERED_WINDOW_THREAD_FINISHED;
			break;
		}
		if (bufferedWindow->bufferManager == NULL) continue;
		FrameBuffer *cpuHandle = ((FrameBufferManager*)bufferedWindow->bufferManager)->cpuHandle();
		if (cpuHandle == NULL) continue;
		if (deviceSynchNeeded)
			if (!cpuHandle->updateHostBlocks(
				((FrameBufferManager*)bufferedWindow->bufferManager)->gpuHandle(bufferedWindow->deviceId), 0, cpuHandle->getBlockCount())) continue;
		bufferedWindow->window().updateFromHost(*cpuHandle);
		bufferedWindow->numFramesDisplayed++;
	}
}
