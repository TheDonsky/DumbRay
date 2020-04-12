#include "BufferedWindow.cuh"
#include "../Window/WindowsWindow.cuh"
#include <new>


BufferedWindow::BufferedWindow(OptionFlags optionFlags, Window *target, const wchar_t *windowName, FrameBufferManager *frameBufferManager, int renderingDeviceId, int refreshIntervalMilliseconds) {
	options = optionFlags;
	bufferManager = frameBufferManager;
	deviceId = renderingDeviceId;
	state = 0;
	numFramesDisplayed = 0;
	if (target != NULL) {
		targetWindow = target;
		if (windowName != NULL)
			targetWindow->setName(windowName);
	}
	else {
		targetWindow = new WindowsWindow(windowName);
		state |= WINDOW_ALLOCATED_INTERNALLY;
	}
	windowThread = std::thread(bufferedWindowThread, this);
	autoRefreshInterval = refreshIntervalMilliseconds;
	if (autoRefreshInterval > 0)
		refreshThread = std::thread(autoRefreshThread, this);
}
BufferedWindow::~BufferedWindow() { 
	closeWindow();
	if ((state & WINDOW_ALLOCATED_INTERNALLY) != 0)
		delete targetWindow;
}

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

bool BufferedWindow::getWindowResolution(int &width, int &height)const { return targetWindow->getResolution(width, height); }
bool BufferedWindow::windowClosed()const { return targetWindow->closed(); }
void BufferedWindow::closeWindow() {
	while (true) {
		std::lock_guard<std::mutex> guard(bufferLock);
		if ((state & BUFFERED_WINDOW_THREAD_FINISHED) != 0) break;
		state |= BUFFERED_WINDOW_SHOULD_EXIT;
		bufferCond.notify_one();
	}
	{
		std::lock_guard<std::mutex> guard(bufferLock);
		if ((state & WINDOW_DESTROYED) != 0) return;
		state |= WINDOW_DESTROYED;
		targetWindow->close();
	}
	windowThread.join();
	if (autoRefreshInterval > 0) refreshThread.join();
}

size_t BufferedWindow::framesDisplayed()const { return numFramesDisplayed; }

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
		
		bufferedWindow->targetWindow->startUpdate();
		int handleWidth, handleHeight;
		cpuHandle->getSize(&handleWidth, &handleHeight);
		bufferedWindow->targetWindow->setImageResolution(handleWidth, handleHeight);
		for (int j = 0; j < handleHeight; j++)
			for (int i = 0; i < handleWidth; i++) {
				Color color = cpuHandle->getColor(i, j);
				bufferedWindow->targetWindow->setPixel(i, j, color.r, color.g, color.b, color.a);
			}
		bufferedWindow->targetWindow->endUpdate();
		bufferedWindow->numFramesDisplayed++;
	}
}

void BufferedWindow::autoRefreshThread(BufferedWindow *bufferedWindow) {
	while (true) {
		std::this_thread::sleep_for(std::chrono::milliseconds(bufferedWindow->autoRefreshInterval));
		{
			std::lock_guard<std::mutex> guard(bufferedWindow->bufferLock);
			if (bufferedWindow->windowClosed() || ((bufferedWindow->state & BUFFERED_WINDOW_SHOULD_EXIT) != 0)) break;
			if (bufferedWindow->bufferManager == NULL) continue;
			FrameBuffer *cpuHandle = ((FrameBufferManager*)bufferedWindow->bufferManager)->cpuHandle();
			if (cpuHandle == NULL) continue;
			bufferedWindow->targetWindow->startUpdate();
			int handleWidth, handleHeight; 
			cpuHandle->getSize(&handleWidth, &handleHeight);
			bufferedWindow->targetWindow->setImageResolution(handleWidth, handleHeight);
			for (int j = 0; j < handleHeight; j++)
				for (int i = 0; i < handleWidth; i++) {
					Color color = cpuHandle->getColor(i, j);
					bufferedWindow->targetWindow->setPixel(i, j, color.r, color.g, color.b, color.a);
				}
			bufferedWindow->targetWindow->endUpdate();
		}
	}
}
