#pragma once
#include "Window.cuh"
#include "../../../Namespaces/Windows/Windows.h"
#include <stdint.h>


class WindowsWindow : public Window {
public:
	WindowsWindow(const char *windowName);
	virtual ~WindowsWindow();

	virtual void setName(const char *name);

	virtual bool getResolution(int &width, int &height)const;
	virtual bool closed()const;
	virtual void close();

	virtual void startUpdate();
	virtual void setImageResolution(int width, int height);
	virtual void setPixel(int x, int y, float r, float g, float b, float a);
	virtual void endUpdate();


private:
	enum StateFlags {
		STATE_WINDOW_OBJECT_DESTROYED = 1
	};
	uint8_t state;
	Matrix<Color> image;
	char windowMemory[sizeof(Windows::Window)];
	Windows::Window &window();
	const Windows::Window &window()const;
};

