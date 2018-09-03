#include "ImageWindow.h"



ImageWindow::ImageWindow(ImageUpdated callback, void *callbackAux) {
	flags = 0;
	imageUpdated = callback;
	aux = callbackAux;
	w = h = 0;
}


ImageWindow::~ImageWindow() { }

void ImageWindow::setWidth(int width) { w = width; }
void ImageWindow::setHeight(int height) { h = height; }

void ImageWindow::setName(const char *name) { }

bool ImageWindow::getResolution(int &width, int &height)const {
	width = w;
	height = h;
	return true;
}
bool ImageWindow::closed()const {
	return false;
}
void ImageWindow::close() {
	flagLock.lock();
	flags |= FLAG_CLOSED;
	flagLock.unlock();
}

void ImageWindow::startUpdate() {
	updateLock.lock();
}
void ImageWindow::setImageResolution(int width, int height) {
	resolutionLock.lock();
	image = QImage(width, height, QImage::Format_RGBA8888);
	resolutionLock.unlock();
}
void ImageWindow::setPixel(int x, int y, float r, float g, float b, float a) {
	image.setPixel(x, y, qRgba(
		(int)((std::min(std::max(r, 0.0f), 1.0f)) * 255),
		(int)((std::min(std::max(g, 0.0f), 1.0f)) * 255),
		(int)((std::min(std::max(b, 0.0f), 1.0f)) * 255),
		(int)((std::min(std::max(a, 0.0f), 1.0f)) * 255)));
}
void ImageWindow::endUpdate() {
	QPixmap map = QPixmap::fromImage(image);
	updateLock.unlock();
	imageUpdated(aux, map);
}