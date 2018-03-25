#pragma once
#ifdef _WIN32
#include<windows.h>
#include<tchar.h>
#endif
#include<stdlib.h>
#include<string.h>
#include<mutex>
#include<condition_variable>
#include"../../DataStructures/Primitives/Pure/Color/Color.h"
#include"../../DataStructures/GeneralPurpose/Matrix/Matrix.h"
#include"../../DataStructures/Screen/FrameBuffer/FrameBuffer.cuh"





namespace Windows{


	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
	/** ########################################################################## **/
	/** Window:                                                                    **/
	/** ########################################################################## **/
	class Window{
	public:
		/** ########################################################################## **/
		/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
		/** ########################################################################## **/
		inline Window(const char *windowName = "WindowName", const char *className = "Window Class");
		inline ~Window();





		/** ########################################################################## **/
		/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
		/** ########################################################################## **/
		
		inline int getWidth()const;
		inline int hetHeight()const;
		inline bool getDimensions(int &width, int &height)const;

		inline bool setWidth(int newWidth);
		inline bool setHeight(int newHeight);
		inline bool setDimensions(int width, int height);

		inline bool dead()const;
		inline bool inFocus()const;





		/** ########################################################################## **/
		/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
		/** ########################################################################## **/
		inline void updateFromHost(const FrameBuffer &image);
		inline void updateFrameHost(const Matrix<Color> &image);
		inline void updateFrameHost(const Color *devImage, int width, int height);
		//inline void updateFromDevice(const FrameBuffer *image);
		inline void updateFrameDevice(const Matrix<Color> *devImage);
		inline void updateFrameDevice(const Color *devImage, int width, int height);





	private:
		/** ########################################################################## **/
		/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
		/** ########################################################################## **/
#ifdef _WIN32
		HINSTANCE hInstance;
		HWND hwnd;
		volatile bool windowDead;
		std::thread messageThread;
		volatile bool hwndInFocus;
		struct Content{
			inline void init();
			inline void dispose();
			inline bool set(int width, int height);
			inline bool loadFromHost(const FrameBuffer &image);
			inline bool loadFromHost(const Color *image, int width, int height);
			inline bool loadFromDevice(const Color *image, int width, int height);

			HBITMAP bitmap;
			COLORREF *colorHost;
			COLORREF *colorDevice;
			int bitWidth, bitHeight;
		}content;





		/** ########################################################################## **/
		/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
		/** ########################################################################## **/
		inline static LRESULT CALLBACK windowProcedure(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
		inline static void createWindow(Window *thisWindow, const char *windowName, const char *className, volatile bool *status, std::condition_variable *statusCondition);
		inline void display();
#else
		int width, height;
#endif
	};
}



#include"Windows.impl.h"


