#include "WindowsWindow.cuh"



WindowsWindow::WindowsWindow(const char *windowName) {
	state = 0;
	new (&window()) Windows::Window(windowName);
}
WindowsWindow::~WindowsWindow() {
	close();
}

void WindowsWindow::setName(const char *name) { }

bool WindowsWindow::getResolution(int &width, int &height)const {
	return window().getDimensions(width, height);
}
bool WindowsWindow::closed()const {
	return window().dead();
}
void WindowsWindow::close() {
	if (state & STATE_WINDOW_OBJECT_DESTROYED) return;
	state |= STATE_WINDOW_OBJECT_DESTROYED;
	window().~Window();
}

void WindowsWindow::startUpdate() { }
void WindowsWindow::setImageResolution(int width, int height) {
	image.setDimensions(width, height);
}
void WindowsWindow::setPixel(int x, int y, float r, float g, float b, float a) {
	image[y][x] = Color(r, g, b, a);
}
void WindowsWindow::endUpdate() {
	window().updateFrameHost(image);
}

Windows::Window &WindowsWindow::window() {
	return (*((Windows::Window*)windowMemory));
}
const Windows::Window &WindowsWindow::window()const {
	return (*((const Windows::Window*)windowMemory));
}
