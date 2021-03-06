#include"Windows.h"



#ifdef _WIN32
/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
namespace Windows{
	namespace Private{
#define WINDOWS_KERNELS_THREADS_PER_BLOCKS 128
#define WINDOWS_KERNELS_UNITS_PER_THREAD 8
		__device__ __host__ static COLORREF translateColor(const Color &c){
			return RGB(((int)(max(min(c.b, 1.0f), 0.0f) * 255)), ((int)(max(min(c.g, 1.0f), 0.0f) * 255)), ((int)(max(min(c.r, 1.0f), 0.0f) * 255)));
		}

		__global__ static void translate(const Color *source, COLORREF *destination, int dataSize){
			int i = blockIdx.x * blockDim.x * WINDOWS_KERNELS_UNITS_PER_THREAD + threadIdx.x * WINDOWS_KERNELS_UNITS_PER_THREAD;
			int end = i + WINDOWS_KERNELS_UNITS_PER_THREAD;
			if (end > dataSize) end = dataSize;
			while (i < end){
				destination[i] = translateColor(source[i]);
				i++;
			}
		}

		static int numThreads(int){
			return WINDOWS_KERNELS_THREADS_PER_BLOCKS;
		}
		static int numBlocks(int dataSize){
			int unitsPerBlock = WINDOWS_KERNELS_THREADS_PER_BLOCKS * WINDOWS_KERNELS_UNITS_PER_THREAD;
			return ((dataSize + unitsPerBlock - 1) / unitsPerBlock);
		}


		__device__ __host__ static int blockCount(int height){
			return height;
		}
		__device__ __host__ static int unitsPerThread() {
			return 32;
		}
		__device__ __host__ static int threadsPerBlock(int width){
			int unitsPerT = unitsPerThread();
			return ((width + unitsPerT - 1) / unitsPerT);
		}

		__global__ static void translate(const Matrix<Color> *source, COLORREF *destination, int width, int height, int startX, int startY) {
			int y = blockIdx.x + startY;
			if (y >= height || y >= source->height()) return;
			int x = threadIdx.x * unitsPerThread() + startX;
			int endX = x + unitsPerThread();
			if (endX > width) endX = width;
			if (endX > source->width()) endX = source->width();
			while (x < endX) {
				destination[y * width + x] = translateColor(source->operator()(y, x));
				x++;
			}
		}

#undef WINDOWS_KERNELS_THREADS_PER_BLOCKS
#undef WINDOWS_KERNELS_UNITS_PER_THREAD
	}
}





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



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
Windows::Window::Window(const wchar_t *windowName, const char *className){
	windowDead = false;
	hwndInFocus = false;
	volatile bool status = false;
	std::condition_variable c;
	messageThread = std::thread(createWindow, this, windowName, className, &status, &c);
	std::mutex m;
	while (!status) {
		std::unique_lock<std::mutex> lock(m);
		c.wait(lock);
	}
	content.init();
}


Windows::Window::~Window(){
	windowDead = true;
	if (messageThread.joinable())
		messageThread.join();
	content.dispose();
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
int Windows::Window::getWidth()const{
	int w, h;
	getDimensions(w, h);
	return(w);
}
int Windows::Window::hetHeight()const{
	int w, h;
	getDimensions(w, h);
	return(h);
}
bool Windows::Window::getDimensions(int &width, int &height)const{
	RECT rect;
	if (GetWindowRect(hwnd, &rect)){
		width = rect.right - rect.left;
		height = rect.bottom - rect.top;
		return true;
	}
	else{
		width = 0;
		height = 0;
		return false;
	}
}

bool Windows::Window::setWidth(int newWidth){
	//
	if (newWidth < 0) newWidth = 0;
	return false;
}
bool Windows::Window::setHeight(int newHeight){
	//
	if (newHeight < 0) newHeight = 0;
	return false;
}

bool Windows::Window::dead()const{
	return windowDead;
}
bool Windows::Window::inFocus()const{
	return hwndInFocus;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
void Windows::Window::updateFromHost(const FrameBuffer &image) {
	if (windowDead) return;
	if (!content.loadFromHost(image)) return;
	display();
}
void Windows::Window::updateFrameHost(const Matrix<Color> &image){
	updateFrameHost(image[0], image.width(), image.height());
}
void Windows::Window::updateFrameHost(const Color *devImage, int width, int height){
	if (windowDead) return;
	if (!content.loadFromHost(devImage, width, height)) return;
	display();
}
void Windows::Window::updateFrameDevice(const Matrix<Color> *devImage) {
	int width, height;
	if (!getDimensions(width, height)) return;
	if (!content.set(width, height)) return;
	cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return;
	Private::translate<<<Private::blockCount(height), Private::threadsPerBlock(width), 0, stream>>>(devImage, content.colorDevice, width, height, 0, 0);
	bool success = (cudaStreamSynchronize(stream) == cudaSuccess);
	if (success) success = (cudaMemcpyAsync(content.colorHost, content.colorDevice, (width * height) * sizeof(COLORREF), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
	if (success) success = (cudaStreamSynchronize(stream) == cudaSuccess);
	if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	if (!success) return;
	if (SetBitmapBits(content.bitmap, sizeof(COLORREF) * width * height, content.colorHost) != 0)
		display();
}
void Windows::Window::updateFrameDevice(const Color *devImage, int width, int height){
	if (windowDead) return;
	if (!content.loadFromDevice(devImage, width, height)) return;
	display();
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
LRESULT CALLBACK Windows::Window::windowProcedure(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam){
	switch (msg){
	case WM_CLOSE:
		DestroyWindow(hwnd);
		break;
	case WM_DESTROY:
		PostQuitMessage(0);
		break;
	default:
		return DefWindowProc(hwnd, msg, wParam, lParam);
	}
	return 0;
}

void Windows::Window::createWindow(Window *thisWindow, const wchar_t *windowName, const char *className, volatile bool *status, std::condition_variable *statusCondition){
	thisWindow->hInstance = GetModuleHandle(NULL);

	WNDCLASS wc = {};

	wc.lpfnWndProc = windowProcedure;
	wc.hInstance = thisWindow->hInstance;
	wc.lpszClassName = (LPCWSTR)className;

	RegisterClass(&wc);

	thisWindow->hwnd = CreateWindowEx(
		0,							// Optional window styles.
		(LPCWSTR) className,		// Window class
		(LPCWSTR) windowName,		// Window text
		WS_OVERLAPPEDWINDOW,		// Window style

		// Size and position
		CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT, CW_USEDEFAULT,

		NULL,					// Parent window    
		NULL,					// Menu
		thisWindow->hInstance,  // Instance handle
		NULL					// Additional application data
		);

	(*status) = true;
	statusCondition->notify_one();

	if (thisWindow->hwnd == NULL) return;

	ShowWindow(thisWindow->hwnd, SW_SHOW);



	MSG Msg;
	while (GetMessage(&Msg, NULL, 0, 0) > 0){
		TranslateMessage(&Msg);
		DispatchMessage(&Msg);
		if (thisWindow->windowDead)
			DestroyWindow(thisWindow->hwnd);
		thisWindow->hwndInFocus = (GetFocus() == thisWindow->hwnd);
	}

	thisWindow->windowDead = true;
}
void Windows::Window::display(){
	HDC hdc = GetDC(hwnd);
	// Temp HDC to copy picture
	HDC src = CreateCompatibleDC(hdc); // hdc - Device context for window, I've got earlier with GetDC(hWnd) or GetDC(NULL);
	SelectObject(src, content.bitmap); // Inserting picture into our temp HDC
	// Copy image from temp HDC to window
	BitBlt(
		hdc,				// Destination
		0,					// x
		0,					// y
		content.bitWidth,	// width
		content.bitHeight,	// height
		src,				// source
		0,					// x and
		0,					// y of upper left corner  of part of the source, from where we'd like to copy
		SRCCOPY);			// Defined DWORD to juct copy pixels. Watch more on msdn;

	DeleteDC(src);	// Deleting temp HDC

	ReleaseDC(hwnd, hdc);	// Release the DC
	UpdateWindow(hwnd);		// Update
}







void Windows::Window::Content::init(){
	bitmap = NULL;
	colorHost = NULL;
	colorDevice = NULL;
	bitWidth = 0;
	bitHeight = 0;
}
void Windows::Window::Content::dispose(){
	if (bitmap != NULL) DeleteObject(bitmap);
	if (colorHost != NULL) delete[] colorHost;
	if (colorDevice != NULL) cudaFree(colorDevice);
	init();
}
bool Windows::Window::Content::set(int width, int height){
	if (bitWidth != width || bitHeight != height || bitmap == NULL || colorHost == NULL || colorDevice == NULL){
		dispose();
		colorHost = new COLORREF[width * height];
		bool deviceSet;
		{
			int cudaDevice;
			deviceSet = (cudaGetDevice(&cudaDevice) == cudaSuccess);
			if (deviceSet) cudaMalloc(&colorDevice, sizeof(COLORREF) * width * height);
		}
		bitmap = CreateBitmap(
			width,						// width.
			height,						// height
			1,							// Color Planes, unfortanutelly don't know what is it actually. Let it be 1
			8 * sizeof(COLORREF),		// Size of memory for one pixel in bits (in win32 4 bytes = 4*8 bits)
			NULL);						// pointer to array
		if (bitmap == NULL || colorHost == NULL || (deviceSet && colorDevice == NULL)){
			dispose();
			return false;
		}
		bitWidth = width;
		bitHeight = height;
	}
	return true;
}
bool Windows::Window::Content::loadFromHost(const FrameBuffer &image) {
	int width, height;
	image.getSize(&width, &height);
	if (!set(width, height)) return false;
	for (int j = 0; j < height; j++)
		for (int i = 0; i < width; i++)
			colorHost[(j * width) + i] = Private::translateColor(image.getColor(i, j));
	return (SetBitmapBits(bitmap, sizeof(COLORREF) * (width * height), colorHost) != 0);
}
bool Windows::Window::Content::loadFromHost(const Color *image, int width, int height){
	if (!set(width, height)) return false;
	int surface = width * height;
	for (int i = 0; i < surface; i++)
		colorHost[i] = Private::translateColor(image[i]);
	return (SetBitmapBits(bitmap, sizeof(COLORREF) * surface, colorHost) != 0);
}
bool Windows::Window::Content::loadFromDevice(const Color *image, int width, int height){
	if (!set(width, height)) return false;
	if (colorDevice == NULL) return false;
	cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
	int surface = width * height;
	Private::translate<<<Private::numBlocks(surface), Private::numThreads(surface), 0, stream>>>(image, colorDevice, surface);
	bool success = (cudaStreamSynchronize(stream) == cudaSuccess);
	if (success) success = (cudaMemcpyAsync(colorHost, colorDevice, surface * sizeof(COLORREF), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
	if (success) success = (cudaStreamSynchronize(stream) == cudaSuccess);
	if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	if (!success) return false;
	return (SetBitmapBits(bitmap, sizeof(COLORREF) * width * height, colorHost) != 0);
}
#else

/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
Windows::Window::Window(const char *windowName, const char *className) { width = 1280; height = 512; }
Windows::Window::~Window() {}
/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
int Windows::Window::getWidth()const { return width; }
int Windows::Window::hetHeight()const { return height; }
bool Windows::Window::getDimensions(int &width, int &height)const { width = this->width; height = this->height; return true; }
bool Windows::Window::setWidth(int newWidth) { width = newWidth; return true; }
bool Windows::Window::setHeight(int newHeight) { height = newHeight; return true; }
bool Windows::Window::setDimensions(int width, int height) { this->width = width; this->height = height; return true; }
bool Windows::Window::dead()const { return false; }
bool Windows::Window::inFocus()const { return false; }
/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
void Windows::Window::updateFromHost(const FrameBuffer &image) {}
void Windows::Window::updateFrameHost(const Matrix<Color> &image) {}
void Windows::Window::updateFrameHost(const Color *devImage, int width, int height) {}
void Windows::Window::updateFrameDevice(const Matrix<Color> *devImage) {}
void Windows::Window::updateFrameDevice(const Color *devImage, int width, int height) {}
#endif
