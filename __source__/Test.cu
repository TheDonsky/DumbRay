#include"DataStructures/GeneralPurpose/Stacktor/Stacktor.test.cuh"
#include"DataStructures/GeneralPurpose/IntMap/IntMap.test.h"
#include"DataStructures/GeneralPurpose/Cutex/Cutex.test.cuh"
#include"DataStructures/Objects/Scene/Raycasters/Octree/Octree.test.cuh"
#include"DataStructures/GeneralPurpose/Generic/Generic.test.cuh"
#include"DataStructures/Objects/Components/Lenses/Lense.test.cuh"
//#include"DataStructures/Renderers/BackwardTracer/BackwardTracer.test.cuh"
#include"DataStructures/GeneralPurpose/Handler/Handler.test.cuh"
#include"DataStructures/Objects/Components/Shaders/Material.test.cuh"
#include"DataStructures/GeneralPurpose/TypeTools/TypeTools.test.cuh"
#include"DataStructures/Objects/Scene/SceneHandler/SceneHandler.test.cuh"
#include"DataStructures/Renderers/Renderer/Renderer.test.cuh"
#include"DataStructures/Screen/FrameBuffer/MemoryMappedFrameBuffer/MemoryMappedFrameBuffer.test.cuh"
#include"Namespaces/Device/Device.cuh"
#include <map>
#include <string>
#include <ctype.h>
namespace {
	typedef void(*TestFunction)();
	struct TestEntry {
		std::string description;
		TestFunction function;
	};
	static void test() {
		cudaSetDevice(0);
		Device::dumpCurrentDevice();
		std::map<std::string, TestEntry> tests;
		tests["scene_handler"] = { "Basic test for SceneHandler structure", SceneHandlerTest::test };
		tests["renderer"] = { "Basic test for standrad Renderer pipeline", RendererTest::test };
		tests["type_tools"] = { "General tests for TypeTools and it's default implementations", TypeToolsTest::test };
		//tests["backward_tracer"] = { std::string("Test for BackwardTracer\n") 
		//	+ "    (legacy; can and most likely, will cause freeze and/or a crash)", BackwardTracerTest::test };
		tests["handler"] = { "General test for generic Handler type", HandlerTest::test };
		tests["generic"] = { "General test for Generic interface", GenericTest::test };
		tests["lense_test_memory"] = { "Simple test for Lense", LenseTest::testMemory };
		tests["material"] = { "Simple test for material", MaterialTest::test };
		tests["octree"] = { std::string("Combined (CPU/GPU) performance test for Octree\n")
			+ "    (simulates rendering with no bounces and shadows;\n"
			+ "    swaps out device when space is pressed)", OctreeTest::test };
		tests["octree_continuous"] = { "Runs 'octree' test over and over again", OctreeTest::runtContinuousTest };
		tests["cutex"] = { std::string("Basic functionality test for Cutex\n")
			+ "    (cuda-friendly mutex equivalent)", CutexTest::test };
		tests["stacktor_local"] = { "Tests for Stacktor local functionality (no upload/destroy)", StacktorTest::localTest };
		tests["stacktor_load"] = { "Tests for Stacktor upload/destroy functionality", StacktorTest::loadTest };
		tests["stacktor_full"] = { "Full Stacktor test, combining both load and local", StacktorTest::test };
		tests["stacktor"] = { "Short for 'stacktor_full'", StacktorTest::test };
		tests["int_map"] = { "Basic tests for IntMap", IntMapTest::test };
		tests["mmaped_frame_buffer"] = { "Basic test for MemoryMappedFrameBuffer", MemoryMappedFrameBufferTest::test };
		std::cout << "___________________________________________________________________" << std::endl;
		std::cout << "WELCOME TO DumbRay TESTING MODULE" << std::endl << "(enter ? for further instructions or any test to run)" << std::endl;
		while (true) {
			std::cout << "--> ";
			std::string line;
			std::getline(std::cin, line);
			std::string command;
			size_t i = 0;
			while (i < line.length() && isspace(line[i])) i++;
			while (i < line.length() && (!isspace(line[i]))) {
				command += tolower(line[i]);
				i++;
			}
			if (command == "?") {
				std::cout << "___________________________________________________________________" << std::endl;
				std::cout << "AVAILABLE TESTS: " << std::endl;
				for (std::map<std::string, TestEntry>::const_iterator iterator = tests.begin(); iterator != tests.end(); iterator++) {
					std::cout << "____________________________" << std::endl;
					std::cout << iterator->first << ":\n    " << iterator->second.description << std::endl;
				}
				std::cout << "___________________________________________________________________" << std::endl;
				std::cout << "SPECIAL COMMANDS: " << std::endl;
				std::cout << "____________________________" << std::endl;
				std::cout << "?:\n    List available commands" << std::endl;
				std::cout << "____________________________" << std::endl;
				std::cout << "exit:\n    Exit testing module" << std::endl;
				std::cout << std::endl << std::endl;
			}
			else if (command == "exit") break;
			else {
				std::map<std::string, TestEntry>::const_iterator iterator = tests.find(command);
				if (iterator != tests.end()) iterator->second.function();
			}
		}
		cudaDeviceReset();
	}
}
#include"DataStructures/Objects/Scene/Scene.cuh"
#include"DataStructures/Objects/Components/Lenses/DefaultPerspectiveLense/DefaultPerspectiveLense.cuh"
#include"DataStructures/Objects/Scene/Raycasters/ShadedOctree/ShadedOctree.cuh"
#include"DataStructures/Renderers/BackwardRenderer/BackwardRenderer.cuh"
#include"DataStructures/Screen/FrameBuffer/MemoryMappedFrameBuffer/MemoryMappedFrameBuffer.cuh"
#include"DataStructures/Screen/FrameBuffer/FrameBuffer.cuh"
#include"Namespaces/Windows/Windows.h"
#include"DataStructures/Objects/Scene/Lights/SimpleDirectionalLight/SimpleDirectionalLight.cuh"
#include<time.h>
#include<iomanip>
void testBackwardRenderer() {
	std::cout << "Concurrent blocks: " << Device::multiprocessorCount() << std::endl;
	while (true) {
		FrameBufferManager frameBuffer;
		frameBuffer.cpuHandle()->use<MemoryMappedFrameBuffer>();
		frameBuffer.cpuHandle()->setResolution(1920, 1080);
		{
			Scene<BakedTriFace> scene;
			scene.lights.flush(1);
			Vector3 direction = Vector3(0.2f, -0.4f, 0.7f).normalized();
			scene.lights[0].use<SimpleDirectionalLight>(
				Photon(Ray(-direction * 10000.0f, direction), 
					Color(1.0f, 1.0f, 1.0f)));
			scene.cameras.flush(1);
			scene.cameras[0].transform.setPosition(Vector3(0, 0, -128));
			scene.cameras[0].lense.use<DefaultPerspectiveLense>(60.0f);
			SceneHandler<BakedTriFace> sceneHandler(scene);
			BackwardRenderer<BakedTriFace>::Configuration configuration(sceneHandler);
			BackwardRenderer<BakedTriFace> renderer(configuration,
				Renderer::ThreadConfiguration::cpuOnly());
			renderer.setFrameBuffer(frameBuffer);
			Windows::Window window;
			const int n = 256;
			std::cout <<
				"__________________________________________" << std::endl
				<< "WAIT...";
			clock_t start = clock();
			for (int i = 0; i < n; i++) {
				if (i > 0 && i % 32 == 0)
					std::cout << ".";
				//*
				if (i % 16 == 0) {
					window.updateFrameHost(
						frameBuffer.cpuHandle()->getData(), 1920, 1080);
					renderer.resetIterations();
					memset(frameBuffer.cpuHandle()->getData(), 0, 1920 * 1080 * sizeof(Color));
				}
				//*/
				renderer.iterate();
			}
			clock_t deltaTime = (clock() - start);
			double time = (((double)deltaTime) / CLOCKS_PER_SEC);
			double iterationClock = (((double)deltaTime) / n);
			double iterationTime = (time / n);
			std::cout << std::fixed << std::setprecision(8) << std::endl <<
				"ITERATIONS:            " << n << std::endl <<
				"TOTAL CLOCK:           " << deltaTime << std::endl <<
				"TOTAL TIME:            " << time << "sec" << std::endl <<
				"ITERATION CLOCK:       " << iterationClock << std::endl <<
				"ITERATION TIME:        " << iterationTime << "sec" << std::endl <<
				"ITERATIONS PER SECOND: " << (1.0 / iterationTime) << std::endl;
		}
		std::string s;
		std::cout << "ENTER ANYTHING TO QUIT... ";
		std::getline(std::cin, s);
		if (s.length() > 0) break;
	}
}


namespace {

#define HOR_SAMPLES 8
#define VER_SAMPLES 8

	__device__ __host__ inline void renderBlock(Color *color, int width, int height, int threadWidth, int threadHeight, int frameId, bool checkerboard, int blockX, int threadX) {
		int xStart = (blockX * threadWidth), yStart = (threadX * threadHeight);
		int xEnd = min((xStart + threadWidth), width), yEnd = min((yStart + threadHeight), height);
		float frameVal = (((float)frameId) / 256.0f);
		register int dj = (checkerboard ? 2 : 1);
		for (int i = yStart; i < yEnd; i++) {
			Color *line = (color + (width * i));
			for (int j = xStart + (checkerboard ? (i & 1) : 0); j < xEnd; j += dj) {
				Color c(0, 0, 0);
				for (float y = i; y < (i + 1); y += (1.0f / VER_SAMPLES))
					for (float x = j; x < (j + 1); x += (1.0f / HOR_SAMPLES))
						c += Color(
							cos(sin((y * x / frameVal) / 1024.0f + frameVal) + frameVal),
							sin(tan(sin((x / y * frameVal) / 1024.0f + frameVal)) * (frameVal + x + y) / 128.0f),
							cos(tan((x + y * frameVal) / 1024.0f * frameVal)) * frameVal);
				line[j] = (c / (HOR_SAMPLES * VER_SAMPLES));
			}
		}
	}
	__device__ __host__ inline void checkerboardFixBlock(Color *color, int width, int height, int threadWidth, int threadHeight, int blockX, int threadX) {
		int xStart = (blockX * threadWidth), yStart = (threadX * threadHeight);
		int xEnd = min((xStart + threadWidth), width), yEnd = min((yStart + threadHeight), height);
		for (int i = yStart; i < yEnd; i++) {
			Color *line = (color + (width * i));
			for (int j = xStart + (((i & 1) == 0) ? 1 : 0); j < xEnd; j += 2) {
				Color c(0, 0, 0);
				float count = 0;
				if (j > 0) { c += line[j - 1]; count++; }
				if (j < (width - 1)) { c += line[j + 1]; count++; }
				if (i > 0) { c += (line - 1)[j]; count++; }
				if (i < (height - 1)) { c += (line + 1)[j]; count++; }
				if (count > 0) c /= count;
				line[j] = c;
			}
		}
	}

	__global__ void render(Color *color, int width, int height, int threadWidth, int threadHeight, int frameId, bool checkerboard) {
		renderBlock(color, width, height, threadWidth, threadHeight, frameId, checkerboard, blockIdx.x, threadIdx.x);
	}
	__global__ void checkerboardFix(Color *color, int width, int height, int threadWidth, int threadHeight) {
		checkerboardFixBlock(color, width, height, threadWidth, threadHeight, blockIdx.x, threadIdx.x);
	}


#define CHUNK_COUNT(total, chunkSize) ((total + chunkSize - 1) / chunkSize)

	void testCheckerboardCPUThreadMain(
		Color *color, int width, int height, int threadWidth, int threadHeight, volatile int *frameId, bool checkerboard,
		Semaphore *semaphore, Semaphore *release, std::mutex *blockLock, volatile int *numBlocks, volatile bool *killSwitch) {
		int kernelWidth = CHUNK_COUNT(width, threadWidth);
		while (true) {
			semaphore->wait();
			if (!(*killSwitch)) while (true) {
				blockLock->lock();
				int blockId = ((*numBlocks) - 1);
				(*numBlocks)--;
				blockLock->unlock();
				if (blockId >= 0) {
					int blockY = (blockId / kernelWidth);
					int blockX = (blockId - (blockY * kernelWidth));
					renderBlock(color, width, height, threadWidth, threadHeight, *frameId, checkerboard, blockX, blockY);
				}
				else break;
			}
			release->post();
			if (*killSwitch) break;
		}
	}
	void testCheckerboardCPUThreadFix(
		Color *color, int width, int height, int threadWidth, int threadHeight, volatile int *frameId, bool checkerboard,
		Semaphore *semaphore, Semaphore *release, std::mutex *blockLock, volatile int *numBlocks, volatile bool *killSwitch) {
		int kernelWidth = CHUNK_COUNT(width, threadWidth);
		while (true) {
			semaphore->wait();
			if (!(*killSwitch)) while (true) {
				blockLock->lock();
				int blockId = ((*numBlocks) - 1);
				(*numBlocks)--;
				blockLock->unlock();
				if (blockId >= 0) {
					int blockY = (blockId / kernelWidth);
					int blockX = (blockId - (blockY * kernelWidth));
					if (checkerboard) checkerboardFixBlock(color, width, height, threadWidth, threadHeight, blockX, blockY);
				}
				else break;
			}
			release->post();
			if (*killSwitch) break;
		}
	}

	void testCheckerboardCPU(int width, int height, int threadWidth, int threadHeight, bool checkerboard, std::mutex *ioLock) {
		Color *color = new Color[width * height];
		if (color == NULL) { std::cout << "ERROR(testCheckerboardCPU): Buffer allocation failed" << std::endl; return; }
		Windows::Window window((std::string("testCheckerboardCPU") + (checkerboard ? " (checkerboard)" : " (native)")).c_str());
		const int numThreads = std::thread::hardware_concurrency();
		std::thread *threads = new std::thread[numThreads];
		std::thread *fixThreads = new std::thread[numThreads];
		Semaphore *semaphores = new Semaphore[numThreads];
		Semaphore *fixSemaphores = new Semaphore[numThreads];
		Semaphore semaphore;
		std::mutex blockLock;
		volatile int numBlocks;
		int kernelWidth = CHUNK_COUNT(width, threadWidth);
		int kernelHeight = CHUNK_COUNT(height, threadHeight);
		volatile int frameId = 0;
		int lastCheckFrameId = frameId;
		time_t lastCheck = clock();
		volatile bool killSwitch = false;
		for (int i = 0; i < numThreads; i++) {
			threads[i] = std::thread(
				testCheckerboardCPUThreadMain, color, width, height, threadWidth, threadHeight, &frameId, checkerboard,
				semaphores + i, &semaphore, &blockLock, &numBlocks, &killSwitch);
			if (checkerboard) fixThreads[i] = std::thread(
				testCheckerboardCPUThreadFix, color, width, height, threadWidth, threadHeight, &frameId, checkerboard,
				fixSemaphores + i, &semaphore, &blockLock, &numBlocks, &killSwitch);
		}
		while (!window.dead()) {
			numBlocks = (kernelWidth * kernelHeight);
			for (int i = 0; i < numThreads; i++) semaphores[i].post();
			for (int i = 0; i < numThreads; i++) semaphore.wait();
			if (checkerboard) {
				numBlocks = (kernelWidth * kernelHeight);
				for (int i = 0; i < numThreads; i++) fixSemaphores[i].post();
				for (int i = 0; i < numThreads; i++) semaphore.wait();
			}
			window.updateFrameHost(color, width, height);
			frameId++;
			time_t now = clock();
			time_t deltaClock = (now - lastCheck);
			if (deltaClock >= CLOCKS_PER_SEC) {
				ioLock->lock();
				std::cout << "FPS: " << ((frameId - lastCheckFrameId) / (((float)deltaClock) / CLOCKS_PER_SEC)) << " (CPU)" << std::endl;
				ioLock->unlock();
				lastCheck = now;
				lastCheckFrameId = frameId;
			}
		}
		killSwitch = true;
		for (int i = 0; i < numThreads; i++) { semaphores[i].post(); fixSemaphores[i].post(); }
		for (int i = 0; i < numThreads; i++) { threads[i].join(); if(checkerboard) fixThreads[i].join(); }
		delete[] semaphores;
		delete[] fixSemaphores;
		delete[] threads;
		delete[] fixThreads;
		delete[] color;
	}

	void testCheckerboard() {
		cudaStream_t stream[2];
		for (int i = 0; i < 2; i++)
			if (cudaStreamCreate(stream + i) != cudaSuccess) {
				std::cout << "ERROR(testCheckerboard): Cuda stream creation failed...." << std::endl;
				return;
			}
		int width = 1920, height = 1080;
		int threadWidth = 4, threadHeight = 4;
		Color *color[2] = { NULL, NULL };
		for (int i = 0; i < 2; i++)
			if (cudaMalloc(color + i, sizeof(Color) * width * height) != cudaSuccess) {
				std::cout << "ERROR(testCheckerboard): Color buffer allocation failed...." << std::endl;
				cudaStreamDestroy(stream[0]); cudaStreamDestroy(stream[1]);
				return;
			}
		Windows::Window checkerboard("testCheckerboard window (checkerboard)");
		Windows::Window native("testCheckerboard window (native)");
		std::mutex ioLock;
		std::thread cpuRenderer(testCheckerboardCPU, width, height, threadWidth, threadHeight, true, &ioLock);
		int kernelWidth = CHUNK_COUNT(width, threadWidth);
		int kernelHeight = CHUNK_COUNT(height, threadHeight);
		int frameId = 0;
		int lastCheckFrameId = frameId;
		time_t lastCheck = clock();
		while ((!native.dead()) || (!checkerboard.dead())) {
			if (!native.dead()) render<<<kernelWidth, kernelHeight, 0, stream[0]>>>(color[0], width, height, threadWidth, threadHeight, frameId, false);
			if (!checkerboard.dead()) {
				render<<<kernelWidth, kernelHeight, 0, stream[1]>>>(color[1], width, height, threadWidth, threadHeight, frameId, true);
				checkerboardFix<<<kernelWidth, kernelHeight, 0, stream[1]>>>(color[1], width, height, threadWidth, threadHeight);
			}
			bool success = (cudaStreamSynchronize(stream[0]) == cudaSuccess);
			if (!(cudaStreamSynchronize(stream[1]) == cudaSuccess)) success = false;
			if (!success) {
				std::cout << "ERROR(testCheckerboard): Stream synchronisation failed...." << std::endl;
				cudaStreamDestroy(stream[0]); cudaStreamDestroy(stream[1]);
				cudaFree(color[0]); cudaFree(color[1]);
				return;
			}
			frameId++;
			time_t now = clock();
			time_t deltaClock = (now - lastCheck);
			if (deltaClock >= CLOCKS_PER_SEC) {
				ioLock.lock();
				std::cout << "FPS: " << ((frameId - lastCheckFrameId) / (((float)deltaClock) / CLOCKS_PER_SEC)) << " (GPU)" << std::endl;
				ioLock.unlock();
				lastCheck = now;
				lastCheckFrameId = frameId;
			}
			native.updateFrameDevice(color[0], width, height);
			checkerboard.updateFrameDevice(color[1], width, height);
		}
		cudaStreamDestroy(stream[0]); cudaStreamDestroy(stream[1]);
		cudaFree(color[0]); cudaFree(color[1]); 
		cudaFree(color);
		cpuRenderer.join();
	}
}

int main() {
	Transform trans(Vector3(0, 0, 0), Vector3(0, 0, 0), Vector3(1, 1, 1));
	std::cout << "(-1, 1, 1)" << trans.ray(Vector3(-1, 1, 1)) << std::endl;
	std::cout << "(1, 1, 1)" << trans.ray(Vector3(1, 1, 1)) << std::endl;

	testCheckerboard();
	std::cout << std::fixed << std::setprecision(4);
	//testBackwardRenderer();
	test();
	return 0;
}
