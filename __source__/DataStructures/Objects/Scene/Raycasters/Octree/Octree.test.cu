#include"Octree.test.cuh"
#include"Octree.cuh"
#include"../../../../../Namespaces/MeshReader/MeshReader.test.h"
#include"../../../../../Namespaces/Tests/Tests.h"
#include"../../../../../Namespaces/Windows/Windows.h"
#include"../../../Components/Transform/Transform.h"
#include"../../../Components/Shaders/DefaultShader/DefaultShader.cuh"
#include<iomanip>
#include<thread>
#include<mutex>



namespace OctreeTest {
	namespace Private {
		// ########################################
		// ############# PIXEL COLOR: #############
		// ########################################
		//#define USE_NORMAL_COLOR
		__device__ __host__ inline static void colorPixel(const Octree<Renderable<BakedTriFace> > &octree, Color &pixel, const Transform &trans, int i, int j, int width, int height, int /*frame*/) {
			Octree<Renderable<BakedTriFace> >::RaycastHit hit;
			Vector3 dir((float)(j - width / 2), (float)(height / 2 - i), (float)(width / 2));
			Ray r = trans.ray(dir.normalized());
			if (octree.cast(r, hit)) {
#ifdef USE_NORMAL_COLOR
				Vector3 normal = (hit.object.norm.massCenter(hit.object.vert.getMases(hit.hitPoint)).normalized() >> trans) / 2.0f + Vector3(0.5f, 0.5f, 0.5f);
				normal = Vector3(normal.x, normal.y, 1.0f - normal.z) * (64.0f / max(hit.hitDistance, 64.0f));
				pixel(normal.x, normal.y, normal.z, 1.0f);
#else
				DefaultShader shad;
				/*
				Photon ph(Ray(hit.hitPoint - Vector3(0, 0, 1000000000), Vector3::front()), ColorRGB(1, 1, 1));
				/*/
				Photon ph(trans.frontRay(), ColorRGB(1, 1, 1));
				//*/
				pixel = shad.cast(DefaultShader::ShaderHitInfo { &hit.object->object, ph, hit.hitPoint, r.origin }).observed.color;
#endif
			}
			else {
#ifdef USE_NORMAL_COLOR
				pixel((((i ^ j) + frame) % 256) / 256.0f, (((i & j) - frame) % 256) / 256.0f, (((i | j) + frame) % 256) / 256.0f);
#else
				pixel(0, 0, 0);
#endif
			}
		}

		// ########################################
		// ####### DEVICE DATA DUMP KERNEL: #######
		// ########################################
		/*
		__global__ static void dumpDeviceOctree(const Octree<> *octree) {
			octree->dump();
		}
		//*/

#define OCTREE_TEST_KERNELS_BLOCK_WIDTH 8
#define OCTREE_TEST_KERNELS_BLOCK_HEIGHT 16
		// ########################################
		// ### DEVICE RENDER KERNEL DIMENSIONS: ###
		// ########################################
		__device__ __host__ inline static int numThreads() {
			return (OCTREE_TEST_KERNELS_BLOCK_WIDTH * OCTREE_TEST_KERNELS_BLOCK_HEIGHT);
		}
		__device__ __host__ inline static int numBlocksWidth(int width) {
			return ((width + OCTREE_TEST_KERNELS_BLOCK_WIDTH - 1) / OCTREE_TEST_KERNELS_BLOCK_WIDTH);
		}
		__device__ __host__ inline static int numBlocksHeight(int height) {
			return ((height + OCTREE_TEST_KERNELS_BLOCK_HEIGHT - 1) / OCTREE_TEST_KERNELS_BLOCK_HEIGHT);
		}
		__device__ __host__ inline static int numBlocks(int width, int height) {
			return (numBlocksWidth(width) * numBlocksHeight(height));
		}

		// ########################################
		// ######### DEVICE RENDER KERNEL: ########
		// ########################################
		__global__ static void color(const Octree<Renderable<BakedTriFace> > *octree, Color *image, const Transform trans, int width, int height, int frame) {
			/*
			// This should not compile:
			Octree<> oct = Octree<>();
			Octree<> oct1 = oct;
			Octree<> oct2(oct1);
			//*/

			register int blocksWidth = numBlocksWidth(width);
			register int lineId = (blockIdx.x / blocksWidth);
			register int columnId = (blockIdx.x - (lineId * blocksWidth));

			register int threadLine = (threadIdx.x / OCTREE_TEST_KERNELS_BLOCK_WIDTH);
			register int threadColumn = (threadIdx.x - (threadLine * OCTREE_TEST_KERNELS_BLOCK_WIDTH));

			register int x = columnId * OCTREE_TEST_KERNELS_BLOCK_WIDTH + threadColumn;
			register int y = lineId * OCTREE_TEST_KERNELS_BLOCK_HEIGHT + threadLine;
			if (x < width && y < height) colorPixel(*octree, image[y * width + x], trans, y, x, width, height, frame);
		}
#undef OCTREE_TEST_KERNELS_BLOCK_WIDTH
#undef OCTREE_TEST_KERNELS_BLOCK_HEIGHT

		// ########################################
		// ####### MACRO CONFIGURATION DUMP: ######
		// ########################################
		inline static void dumpConfiguration() {
			std::cout << "########### DUMPING OCTREE COMPILE PARAMETERS ##########" << std::endl;
			std::cout << "OCTREE_POLYCOUNT_TO_SPLIT_NODE: " << OCTREE_POLYCOUNT_TO_SPLIT_NODE << std::endl;
			std::cout << "OCTREE_VOXEL_LOCAL_CAPACITY: " << OCTREE_VOXEL_LOCAL_CAPACITY << std::endl;
			std::cout << "OCTREE_MAX_DEPTH: " << OCTREE_MAX_DEPTH << std::endl;
			std::cout << std::endl;
		}

		// ########################################
		// ######## OCTREE CONSTRUCTION: ##########
		// ########################################
		inline static Octree<Renderable<BakedTriFace> > constructOctree(const Stacktor<PolyMesh> &meshes, Octree<Renderable<BakedTriFace> > *&devOctree) {
			Octree<Renderable<BakedTriFace> > octree;
			Vertex minVert = meshes[0].vertex(0);
			Vertex maxVert = meshes[0].vertex(0);
			for (int i = 0; i < meshes.size(); i++)
				for (int j = 0; j < meshes[i].vertextCount(); j++) {
					if (meshes[i].vertex(j).x < minVert.x) minVert.x = meshes[i].vertex(j).x;
					else if (meshes[i].vertex(j).x > maxVert.x) maxVert.x = meshes[i].vertex(j).x;

					if (meshes[i].vertex(j).y < minVert.y) minVert.y = meshes[i].vertex(j).y;
					else if (meshes[i].vertex(j).y > maxVert.y) maxVert.y = meshes[i].vertex(j).y;

					if (meshes[i].vertex(j).z < minVert.z) minVert.z = meshes[i].vertex(j).z;
					else if (meshes[i].vertex(j).z > maxVert.z) maxVert.z = meshes[i].vertex(j).z;
				}
			octree.reinit(AABB(minVert - EPSILON_VECTOR, maxVert + EPSILON_VECTOR));
			octree.reset();
			for (int i = 0; i < meshes.size(); i++) {
				BakedTriMesh mesh;
				meshes[i].bake(mesh);
				//*
				for (int j = 0; j < mesh.size(); j++)
					octree.push(Renderable<BakedTriFace>(mesh[j], 0));
				std::cout << "PUSHED " << i << std::endl;
				/*/
				octree.put(mesh);
				std::cout << "PUT " << i << std::endl;
				//*/
			}
			//*
			std::cout << "BUILDING..." << std::endl;
			octree.build();
			/*/
			std::cout << "OPTIMIZING..." << std::endl;
			octree.reduceNodes();
			//*/
			std::cout << "UPLOADING..." << std::endl;
			devOctree = octree.upload();
			if (devOctree != NULL) std::cout << "OCTREE UPLOADED" << std::endl;
			else std::cout << "OCTREE UPLOAD FAILED" << std::endl;
			//Octree<Renderable<BakedTriFace> > tmpClone = octree;
			return octree;
		}


		class OctreeTestRenderContext {
		private:
			Octree<Renderable<BakedTriFace> > octree, *devOctree;

			char windowData[sizeof(Windows::Window)];
			Matrix<Color> *image, *imageBack;
			Color *devColor, *devColorBack;
			int devColWidth, devColHeight;
			std::mutex colorLock;
			std::condition_variable colorLockWait;
			volatile bool frameReady;

			bool onDevice, spacePressed;

#ifdef _WIN32
			POINT cursor;
#endif
			Vector3 euler;
			Transform trans;
			Vector3 pivot;
			float distance;
			bool mouseWasDown;

			cudaStream_t stream;
			bool canRunOnCuda;

			struct CPUrenderThread {
				std::thread thread;
				std::condition_variable wait;
				std::condition_variable realease;
				volatile bool b;
			};
			CPUrenderThread *cpuThreads;
			int cpuThreadCount;

			inline Windows::Window& window() {
				return (*((Windows::Window *)windowData));
			}


			// ########################################
			// ############ DEVICE SWITCH: ############
			// ########################################
			inline void switchDevice() {
				if (
#ifdef _WIN32
					GetAsyncKeyState(VK_SPACE) & 0x8000
#else
					false
#endif
					) {
					if (!spacePressed) {
						std::cout << "Changing state...";
						colorLock.lock();
						onDevice = ((!onDevice) && canRunOnCuda);
						colorLock.unlock();
						if (onDevice) std::cout << "USING DEVICE" << std::endl;
						else std::cout << "USING HOST" << std::endl;
					}
					spacePressed = true;
				}
				else spacePressed = false;
			}

			// ########################################
			// ############### ROTATION: ##############
			// ########################################
			inline void rotate() {
				if (window().inFocus()) {
					if (
#ifdef _WIN32
						GetKeyState(VK_LBUTTON) & 0x100
#else
						false
#endif
						) {
#ifdef _WIN32						
						POINT newCursor; GetCursorPos(&newCursor);
						if (mouseWasDown) {
							euler.y += (newCursor.x - cursor.x) / 4.0f;
							euler.x += (newCursor.y - cursor.y) / 4.0f;
							if (euler.x <= -80) euler.x = -80;
							else if (euler.x >= 80) euler.x = 80;
							trans.setEulerAngles(euler);
						}
						else mouseWasDown = true;
						cursor = newCursor;
#endif
					}
					else mouseWasDown = false;
				}
				else mouseWasDown = false;
				trans.setPosition(pivot + trans.back() * distance);
			}

			// ########################################
			// ######## DEVICE RENDER ROUTINE: ########
			// ########################################
			inline bool renderOnDevice(int width, int height, int frame) {
				if (devColWidth != width || devColHeight != height || devColor == NULL) {
					if (devColor != NULL) cudaFree(devColor);
					if (cudaMalloc(&devColor, sizeof(Color) * max(1, width * height)) != cudaSuccess) {
						std::cout << "CUDA_MALLOC PROBLEM" << std::endl;
						return false;
					}
					colorLock.lock();
					if (devColorBack != NULL) cudaFree(devColorBack);
					if (cudaMalloc(&devColorBack, sizeof(Color) * max(1, width * height)) != cudaSuccess) {
						std::cout << "CUDA_MALLOC PROBLEM" << std::endl;
						return false;
					}
					colorLock.unlock();
					devColWidth = width;
					devColHeight = height;
				}
				color << <numBlocks(devColWidth, devColHeight), numThreads(), 0, stream >> >(devOctree, devColor, trans, devColWidth, devColHeight, frame);
				bool success = (cudaStreamSynchronize(stream) == cudaSuccess);
				if (!success) { std::cout << "STREAM JOIN ERROR" << std::endl; return false; }
				if (colorLock.try_lock()) {
					Color *tmp = devColor;
					devColor = devColorBack;
					devColorBack = tmp;
					colorLock.unlock();
					colorLockWait.notify_all();
				}
				return true;
			}

			// ########################################
			// ######### HOST RENDER ROUTINE: #########
			// ########################################
			struct CpuRenderThreadParams {
				const Octree<Renderable<BakedTriFace> > *octree;
				Matrix<Color> *image;
				Transform trans;
				int step;
				int startI;
				int frame;
			};

			inline static void cpuRenderThread(CpuRenderThreadParams params) {
				/*
				const int width = image->width();
				const int height = image->height();
				const int chunkWidth = 32;
				const int chunkHeight = 16;
				const int horChunks = ((width + chunkWidth - 1) / chunkWidth);
				const int verChunks = ((height + chunkHeight - 1) / chunkHeight);
				const int totalChunks = (horChunks * verChunks);
				for (int chunkId = startI; chunkId < totalChunks; chunkId += step) {
					const int chunkY = (chunkId / horChunks);
					const int chunkX = (chunkId - (chunkY * horChunks));
					const int endY = min(height, ((chunkY + 1) * chunkHeight));
					const int endX = min(width, ((chunkX + 1) * chunkWidth));
					const int startX = (chunkX * chunkWidth);
					for (int i = chunkY * chunkHeight; i < endY; i++)
						for (int j = startX; j < endX; j++)
							colorPixel(*octree, image->operator[](i)[j], trans, i, j, width, height, frame);
				}
				/*/
				for (int i = params.startI; i < params.image->height(); i += params.step)
					for (int j = 0; j < params.image->width(); j++)
						colorPixel(*params.octree, params.image->operator[](i)[j], params.trans, i, j, params.image->width(), params.image->height(), params.frame);
				//*/
			}
			inline void renderOnCPU(int width, int height, int frame) {
				if (width != image->width() || height != image->height()) { image->setDimensions(width, height); }
				int numThreads = min(max(std::thread::hardware_concurrency(), 1), 32);
				std::thread threads[32];
				for (int i = 0; i < numThreads; i++) threads[i] = std::thread(cpuRenderThread, CpuRenderThreadParams { &octree, image, trans, numThreads, i, frame });
				for (int i = 0; i < numThreads; i++) threads[i].join();
				if (colorLock.try_lock()) {
					Matrix<Color> *tmp = image;
					image = imageBack;
					imageBack = tmp;
					colorLock.unlock();
					colorLockWait.notify_all();
				}
			}

			// ########################################
			// ############# FRAME RENDER: ############
			// ########################################
			inline bool render(int frame) {
				int width, height;
				window().getDimensions(width, height);
				if (onDevice) return renderOnDevice(width, height, frame);
				else renderOnCPU(width, height, frame);
				return true;
			}

			// ########################################
			// ############### FPS DUMP: ##############
			// ########################################
			inline static void dumpFPS(const int frame, long &time) {
				if (frame % 128 == 0 && frame > 0) {
					long newTime = clock();
					long deltaTime = (newTime - time);
					float avgDeltaTime = ((float)deltaTime) / 128.0f;
					std::cout << "CLOCK: " << avgDeltaTime << " (" << CLOCKS_PER_SEC / avgDeltaTime << " fps)" << std::endl;
					time = newTime;
				}
			}


			inline static void windowUpdateThread(OctreeTestRenderContext *context) {
				while (!context->window().dead()) {
					std::unique_lock<std::mutex> uniqueLock(context->colorLock);
					context->colorLockWait.wait(uniqueLock);
					if (context->onDevice) {
						if (context->devColorBack != NULL);
						context->window().updateFrameDevice(context->devColorBack, context->devColWidth, context->devColHeight);
					}
					else context->window().updateFrameHost(*context->imageBack);
				}
			}
		public:
			// ########################################
			// ######## READING & PREPARATION: ########
			// ########################################
			inline OctreeTestRenderContext() {
				// ################################
				// ############# INTRO: ###########
				std::cout << std::fixed << std::setprecision(2);
				dumpConfiguration();
				// ################################
				// ############# DATA: ############
				Stacktor<PolyMesh> meshes; MeshReaderTest::readMeshes(meshes);
				octree = constructOctree(meshes, devOctree);
				// ############ WINDOW: ###########
				devColor = NULL;
				devColorBack = NULL;
				devColWidth = 0;
				devColHeight = 0;
				frameReady = false;
				image = new Matrix<Color>();
				imageBack = new Matrix<Color>();
				// ############ RENDER: ###########
				std::cout << "READY TO RENDER" << std::endl;
				if (cudaStreamCreate(&stream) != cudaSuccess) { std::cout << "STREAM ALLOCATION ERROR" << std::endl; canRunOnCuda = false; }
				else canRunOnCuda = true;
				onDevice = canRunOnCuda;
				spacePressed = false;
				// ########### ROTATION: ##########
				euler(0, 0, 0);
				mouseWasDown = true;
				// ######### CPU THREADS: #########
				Octree<Vertex> octo;
				octo.put(Vertex(0, 0, 0));
				octo.cast(Ray(Vector3(-32, -32, -32), Vector3(1, 1, 1)));
				// ######### TRANSFORM: #########
				pivot(0, 0, 0);
				distance = 128.0f;
				trans.setEulerAngles(euler);
				trans.setPosition(pivot + trans.back() * distance);
			}

			// ########################################
			// ################ CLEANUP: ##############
			// ########################################
			inline ~OctreeTestRenderContext() {
				if (devOctree != NULL) {
					if (Octree<Renderable<BakedTriFace> >::dispose(devOctree)) std::cout << "DEVICE OCTREE DIPOSED SUCCESSFULY" << std::endl;
					else std::cout << "ERROR DISPOSING OF DEVICE OCTREE" << std::endl;
					cudaFree(devOctree);
				}
				if (devColor != NULL) cudaFree(devColor);
				if (devColorBack != NULL) cudaFree(devColorBack);
				delete image;
				delete imageBack;
				if (canRunOnCuda) if (cudaStreamDestroy(stream) != cudaSuccess) std::cout << "FAILED TO DESTROY STREAM" << std::endl;
			}

			// ########################################
			// ############# RENDER TEST: #############
			// ########################################
			inline void runTest() {
				new(&window()) Windows::Window(L"OCTREE TEST WINDOW");
				std::thread refreshThread(windowUpdateThread, this);
				int frame = 0; long time = clock();
				while (true) {
					if (window().dead()) break;
					switchDevice();
					rotate();
					if (!render(frame)) continue;
					dumpFPS(frame, time);
					frame++;
				}
				colorLockWait.notify_all();
				refreshThread.join();
				window().~Window();
			}
		};

		// ########################################
		// ########## BASIC OCTREE TEST: ##########
		// ########################################
		inline static void test() {
			OctreeTestRenderContext context;
			context.runTest();
		}
	}

	/*
	Tests basic capabilities of Octree by rendering normals with backward ray tracing
	*/
	void test() {
		Tests::runTest(Private::test, "Testing Octree");
	}


	void runtContinuousTest() {
		while (true) {
			std::string s;
			std::cout << "Enter anything to prevent running Octree test... ";
			std::getline(std::cin, s);
			if (s.length() > 0) break;
			OctreeTest::test();
		}
	}
}

