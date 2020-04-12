#pragma once
#include "../__source__/DataStructures/Screen/Window/Window.cuh"
#include <QImage>
#include <QPixmap>
#include <mutex>

class ImageWindow : public Window {
public:
	typedef void(*ImageUpdated)(void *aux, const QPixmap &map);

	ImageWindow(ImageUpdated callback, void *callbackAux);
	~ImageWindow();

	void setWidth(int width);
	void setHeight(int height);

	virtual void setName(const wchar_t *name);

	virtual bool getResolution(int &width, int &height)const;
	virtual bool closed()const;
	virtual void close();

	virtual void startUpdate();
	virtual void setImageResolution(int width, int height);
	virtual void setPixel(int x, int y, float r, float g, float b, float a);
	virtual void endUpdate();



private:
	enum Flags {
		FLAG_CLOSED = 1
	};
	std::mutex flagLock;
	uint8_t flags;

	int w, h;

	std::mutex updateLock;
	mutable std::mutex resolutionLock;
	QImage image;

	ImageUpdated imageUpdated;
	void *aux;
};

