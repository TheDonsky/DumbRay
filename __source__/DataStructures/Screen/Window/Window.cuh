#pragma once



class Window {
public:
	virtual inline ~Window() {}

	virtual void setName(const char *name) = 0;

	virtual bool getResolution(int &width, int &height)const = 0;
	virtual bool closed()const = 0;
	virtual void close() = 0;

	virtual void startUpdate() = 0;
	virtual void setImageResolution(int width, int height) = 0;
	virtual void setPixel(int x, int y, float r, float g, float b, float a) = 0;
	virtual void endUpdate() = 0;
};
