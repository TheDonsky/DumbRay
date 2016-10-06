#include"Stacktor.test.cuh"
#include"PolyMesh.h"
#include"MeshReader.h"
#include"IntMap.test.h"
#include"Windows.h"
#include"Cutex.test.cuh"
#include"Octree.test.cuh"



static void dump(Stacktor<PolyMesh> meshes, Stacktor<String> meshNames){
	std::cout << "####################### OBJECTS READ: #######################" << std::endl;
	for (int i = 0; i < meshes.size(); i++){
		std::cout << "MESH: " << (meshNames[i] + 0) << std::endl;
		std::cout << "   . Verts: " << meshes[i].vertextCount() << std::endl;
		std::cout << "   . Norms: " << meshes[i].normalCount() << std::endl;
		std::cout << "   . Texs:  " << meshes[i].textureCount() << std::endl;
		std::cout << "   . Faces: " << meshes[i].faceCount() << std::endl;
		Tests::logLine();
	}
}

inline static __device__ __host__ Color color(int y, int x, float time){
	//*
	float i = ((float)y) / 4.0f;
	float j = ((float)x) / 4.0f;
	time /= 16.0f;
	Color c;
	c.r = (sin(i * j + time) + 1) / 2.0f;
	c.g = (cos(i / j - time) + 1) / 2.0f;
	c.b = ((int)tan(i + j + time) % 256) / 256.0f;
	c.r = sin(PI * c.r + PI * c.g + PI * c.b);
	c.g = cos(PI * c.r + PI * c.g + PI * c.b);
	c.b = sin(PI * c.r + PI * c.g + PI * c.b) * cos(PI * c.r + PI * c.g + PI * c.b);
	return c;
}

inline static void color(Matrix<Color> &mat, int time){
	time /= 16;
	for (int i = 0; i < mat.height(); i++)
		for (int j = 0; j < mat.width(); j++)
			mat[i][j] = color(i, j, (float)time);
}


#define UNITS_PER_THREAD 8
#define UNITS_PER_BLOCK 4
__global__ static void color(Color *data, int width, int height, int time){
	int y = blockIdx.x * UNITS_PER_BLOCK;
	int startX = threadIdx.x * UNITS_PER_THREAD;
	int endY = y + UNITS_PER_BLOCK; if (endY > height) endY = height;
	int endX = startX + UNITS_PER_THREAD; if (endX > width) endX = width;
	while (y < endY){
		for (int x = startX; x < endX; x++)
			data[y * width + x] = color(y, x, time);
		y++;
	}
}

static int numBlocks(int height){
	return ((height + UNITS_PER_BLOCK - 1) / UNITS_PER_BLOCK);
}
static int numThreads(int width){
	return ((width + UNITS_PER_THREAD - 1) / UNITS_PER_THREAD);
}

static bool colorPixels(Color *data, int width, int height, int time){
	time /= 16;
	cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
	color<<<numBlocks(height), numThreads(width), 0, stream>>>(data, width, height, time);
	bool success = (cudaStreamSynchronize(stream) == cudaSuccess);
	if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	return success;
}

inline static void testWindow(){
	Windows::Window w;
	int i = 0;
	Matrix<Color> mat;
	long time = clock();
	long consumedTime = 0;
	long lastTime = clock();
	
	Color *image = NULL;
	int imgWidth = 0;
	int imgHeight = 0;

	bool device = false;
	short oldState = false;
	while (true){
		if (w.dead()) break;

		int width, height;
		w.getDimensions(width, height);
		if (!device){
			if (mat.width() != width || mat.height() != height){
				mat.setDimensions(width, height);
				continue;
			}
			color(mat, consumedTime);
			w.updateFrameHost(mat);
		}
		else{
			if (imgWidth != width || imgHeight != height || image == NULL){
				if (image != NULL) cudaFree(image);
				if (cudaMalloc(&image, sizeof(Color) * width * height) != cudaSuccess){
					image = NULL;
					std::cout << "ERROR" << std::endl;
					continue;
				}
				imgWidth = width;
				imgHeight = height;
			}
			if (!colorPixels(image, imgWidth, imgHeight, consumedTime))
				std::cout << "ERROR COLORING" << std::endl;
			w.updateFrameDevice(image, imgWidth, imgHeight);
		}

		short state = GetKeyState(VK_SPACE);
		if (state && oldState != state){
			std::cout << "Changing state...";
			device = !device;
			if (device) std::cout << "USING DEVICE" << std::endl;
			else std::cout << "USING HOST" << std::endl;
		}
		oldState = state;

		int deltaTime = (int)(clock() - lastTime);
		lastTime = clock();
		consumedTime += deltaTime;
		if (i % 100 == 0){
			std::cout << "FPS: " << (100.0f * (float)CLOCKS_PER_SEC / (float)(clock() - time)) << std::endl;
			time = clock();
		}
		i++;
	}
}

static void makeAndDestroyWindow(){
	Windows::Window window("I Will Close");
}

static void windowTrap(){
	Windows::Window window("CLOSE ME IF YOU CAN");
	while (!window.dead()) Sleep(256);
	std::thread trap1(windowTrap);
	std::thread trap2(windowTrap);
	trap1.join();
	trap2.join();
}

int main(){

	while (true){
		std::string s;
		std::cout << "Enter anything to prevent running Octree test... ";
		std::getline(std::cin, s);
		if (s.length() > 0) break;
		OctreeTest::test();
	}
	CutexTest::test();

	makeAndDestroyWindow();
	testWindow();

	std::thread windowTrapThread(windowTrap);
	ShowWindow(GetConsoleWindow(), SW_HIDE);

	IntMapTest::test();

	Stacktor<PolyMesh> meshes;
	Stacktor<String> meshNames;

	if (!MeshReader::readObj(meshes, meshNames, "Boxes.obj", true))
		std::cout << "ERROR READING \"Boxes.obj\"" << std::endl;

	long long t;

	t = clock();
	std::cout << "READING \"D:\\Knot.obj\"..." << std::endl;
	if (MeshReader::readObj(meshes, meshNames, "D:\\Knot.obj"))
		std::cout << "CLOCK: " << (clock() - t) << std::endl;
	else std::cout << "FAILED!" << std::endl;

	dump(meshes, meshNames);
	system("PAUSE");

	cudaSetDevice(0);
	Stacktor<PolyMesh> s;
	s.push(PolyMesh());
	s.push(PolyMesh());
	Stacktor<PolyMesh> *arrM; cudaMalloc(&arrM, sizeof(Stacktor<PolyMesh>));
	Stacktor<PolyMesh>::upload(&s, arrM);
	Stacktor<PolyMesh>::dispose(arrM);
	cudaFree(arrM);
	while (true){
		StacktorTest::Load::test();
		StacktorTest::fullTest();
		system("PAUSE");
	}
	windowTrapThread.join();
}