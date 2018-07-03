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
		Window(const char *windowName = "WindowName", const char *className = "Window Class");
		~Window();





		/** ########################################################################## **/
		/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
		/** ########################################################################## **/
		
		int getWidth()const;
		int hetHeight()const;
		bool getDimensions(int &width, int &height)const;

		bool setWidth(int newWidth);
		bool setHeight(int newHeight);
		bool setDimensions(int width, int height);

		bool dead()const;
		bool inFocus()const;





		/** ########################################################################## **/
		/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
		/** ########################################################################## **/
		void updateFromHost(const FrameBuffer &image);
		void updateFrameHost(const Matrix<Color> &image);
		void updateFrameHost(const Color *devImage, int width, int height);
		//void updateFromDevice(const FrameBuffer *image);
		void updateFrameDevice(const Matrix<Color> *devImage);
		void updateFrameDevice(const Color *devImage, int width, int height);





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
			void init();
			void dispose();
			bool set(int width, int height);
			bool loadFromHost(const FrameBuffer &image);
			bool loadFromHost(const Color *image, int width, int height);
			bool loadFromDevice(const Color *image, int width, int height);

			HBITMAP bitmap;
			COLORREF *colorHost;
			COLORREF *colorDevice;
			int bitWidth, bitHeight;
		}content;





		/** ########################################################################## **/
		/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
		/** ########################################################################## **/
		static LRESULT CALLBACK windowProcedure(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam);
		static void createWindow(Window *thisWindow, const char *windowName, const char *className, volatile bool *status, std::condition_variable *statusCondition);
		void display();
#else
		int width, height;
#endif
	};
}




//#include "Windows.impl.cuh"
