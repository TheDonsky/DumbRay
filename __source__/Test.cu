#include"DataStructures/GeneralPurpose/Stacktor/Stacktor.test.cuh"
#include"DataStructures/Objects/Meshes/PolyMesh/PolyMesh.h"
#include"Namespaces/MeshReader/MeshReader.h"
#include"DataStructures/GeneralPurpose/IntMap/IntMap.test.h"
#include"Namespaces/Windows/Windows.h"
#include"DataStructures/GeneralPurpose/Cutex/Cutex.test.cuh"
#include"DataStructures/Objects/Scene/Raycasters/Octree/Octree.test.cuh"
#include"DataStructures/Primitives/Pure/ColorRGB/ColorRGB.cuh"
#include"DataStructures/Objects/Components/Shaders/DummyShader/DummyShader.cuh"
#include"DataStructures/Primitives/Pure/Vector2/Vector2.h"
#include"DataStructures/GeneralPurpose/Generic/Generic.test.cuh"
#include"DataStructures/Objects/Components/Lenses/Lense.test.cuh"
#include"DataStructures/Renderers/BackwardTracer/BackwardTracer.test.cuh"
#include"DataStructures/GeneralPurpose/Handler/Handler.test.cuh"
#include"DataStructures/Objects/Components/Shaders/Material.test.cuh"
#include"DataStructures/GeneralPurpose/TypeTools/TypeTools.test.cuh"
#include"DataStructures/Objects/Scene/Scene.cuh"
#include"DataStructures/Primitives/Compound/Pair/Pair.cuh"
#include"Namespaces/Device/Device.cuh"
#include"DataStructures/Objects/Scene/SceneHandler/SceneHandler.test.cuh"
#include"DataStructures/Renderers/Renderer/Renderer.test.cuh"


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

__dumb__ void printVector(const char *label, const Vector3 &v){
	printf("%s: (%f, %f, %f)\n", label, v.x, v.y, v.z);
}

__global__ void testConstantsKernel(){
	if (blockIdx.x == 0 && threadIdx.x == 0){
		printVector(" zero", Vector3::zero());
		printVector("  one", Vector3::one());
		printVector("   up", Vector3::up());
		printVector(" down", Vector3::down());
		printVector("front", Vector3::front());
		printVector(" back", Vector3::back());
		printVector("right", Vector3::right());
		printVector(" left", Vector3::left());
		printVector("Xaxis", Vector3::Xaxis());
		printVector("Yaxis", Vector3::Yaxis());
		printVector("Zaxis", Vector3::Zaxis());
		printVector("    i", Vector3::i());
		printVector("    j", Vector3::j());
		printVector("    k", Vector3::k());
		printf("\n");
	}
}
static void testConstants() {
	testConstantsKernel<<<1, 1>>>();
	cudaDeviceSynchronize();
}

int main(){
	cudaSetDevice(0);
	Device::dumpCurrentDevice();
	SceneHandlerTest::test();
	//RendererTest::test();
	//TypeToolsTest::test();
	//BackwardTracerTest::test();
	//HandlerTest::test();
	//GenericTest::test();
	//LenseTest::testMemory();
	testConstants();
	MaterialTest::test();
	OctreeTest::runtContinuousTest();
	CutexTest::test();
	StacktorTest::Load::test();
	StacktorTest::fullTest();
	IntMapTest::test();
	makeAndDestroyWindow();
	testWindow();
	cudaDeviceReset();
}
