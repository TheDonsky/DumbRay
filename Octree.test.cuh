#pragma once

#include"Octree.cuh"
#include"MeshReader.h"
#include"Tests.h"
#include"Windows.h"
#include"Transform.h"
#include"DefaultShader.cuh"
#include<iomanip>



namespace OctreeTest{
	namespace Private{
		// ########################################
		// ############# PIXEL COLOR: #############
		// ########################################
//#define USE_NORMAL_COLOR
		__device__ __host__ inline static void colorPixel(const Octree<> &octree, Color &pixel, const Transform &trans, int i, int j, int width, int height, int frame){
			Octree<>::RaycastHit hit;
			Vector3 dir((float)(j - width / 2), (float)(height / 2 - i), (float)(width / 2));
			Ray r = (Ray(Vector3(0, 0, -128.0f), dir.normalized()) << trans);
			if (octree.cast(r, hit)){
#ifdef USE_NORMAL_COLOR
				Vector3 normal = (hit.object.norm.massCenter(hit.object.vert.getMases(hit.hitPoint)).normalized() >> trans) / 2.0f + Vector3(0.5f, 0.5f, 0.5f);
				normal = Vector3(normal.x, normal.y, 1.0f - normal.z) * (64.0f / max(hit.hitDistance, 64.0f));
				pixel(normal.x, normal.y, normal.z, 1.0f);
#else
				DefaultShader shad;
				Photon ph(Ray(hit.hitPoint - Vector3(0, 0, 1000000000), Vector3::front()), ColorRGB(1, 1, 1));
				pixel = shad.cast(Material<BakedTriFace>::HitInfo(hit.object, ph, hit.hitPoint, hit.hitDistance, r.origin)).cameraPhoton.color;
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
		__global__ static void dumpDeviceOctree(const Octree<> *octree){
			octree->dump();
		}

#define OCTREE_TEST_KERNELS_BLOCK_WIDTH 16
#define OCTREE_TEST_KERNELS_BLOCK_HEIGHT 8
		// ########################################
		// ### DEVICE RENDER KERNEL DIMENSIONS: ###
		// ########################################
		__device__ __host__ inline static int numThreads(){
			return (OCTREE_TEST_KERNELS_BLOCK_WIDTH * OCTREE_TEST_KERNELS_BLOCK_HEIGHT);
		}
		__device__ __host__ inline static int numBlocksWidth(int width){
			return ((width + OCTREE_TEST_KERNELS_BLOCK_WIDTH - 1) / OCTREE_TEST_KERNELS_BLOCK_WIDTH);
		}
		__device__ __host__ inline static int numBlocksHeight(int height){
			return ((height + OCTREE_TEST_KERNELS_BLOCK_HEIGHT - 1) / OCTREE_TEST_KERNELS_BLOCK_HEIGHT);
		}
		__device__ __host__ inline static int numBlocks(int width, int height){
			return (numBlocksWidth(width) * numBlocksHeight(height));
		}

		// ########################################
		// ######### DEVICE RENDER KERNEL: ########
		// ########################################
		__global__ static void color(const Octree<> *octree, Color *image, const Transform trans, int width, int height, int frame){
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
		inline static void dumpConfiguration(){
			std::cout << "########### DUMPING OCTREE COMPILE PARAMETERS ##########" << std::endl;
			std::cout << "OCTREE_POLYCOUNT_TO_SPLIT_NODE: " << OCTREE_POLYCOUNT_TO_SPLIT_NODE << std::endl;
			std::cout << "OCTREE_VOXEL_LOCAL_CAPACITY: " << OCTREE_VOXEL_LOCAL_CAPACITY << std::endl;
			std::cout << "OCTREE_MAX_DEPTH: " << OCTREE_MAX_DEPTH << std::endl;
			std::cout << std::endl;
		}

		// ########################################
		// ############ READING FILES: ############
		// ########################################
		inline static void readMeshes(Stacktor<PolyMesh> &meshes){
			Stacktor<String> names;
			while (true){
				std::string line;
				std::cout << "ENTER .obj FILE TO READ(ENTER TO RENDER): ";
				std::getline(std::cin, line);
				if (line == "") break;
				std::cout << "READING FILE... PLEASE WAIT" << std::endl;
				int start = meshes.size();
				if (MeshReader::readObj(meshes, names, line)){
					std::cout << "FILE (" << line << ") CONTENT: " << std::endl;
					for (int i = start; i < meshes.size(); i++)
						std::cout << "    NAME: " << (names[i] + 0) << "; V: " << meshes[i].vertextCount() << "; N: " << meshes[i].normalCount() << "; T: " << meshes[i].textureCount() << "; F: " << meshes[i].faceCount() << std::endl;
					while (true){
						std::cout << "WOULD YOU LIKE TO TRANSFORM THE CONTENT(y/n)? ";
						std::string answer; std::getline(std::cin, answer);
						if (answer == "y" || answer == "Y" || answer == "yes" || answer == "Yes" || answer == "YES"){
							std::cout << "ENTER TRANSFORM POSITION: ";
							Vector3 pos; std::cin >> pos;
							std::cout << "ENTER EULER ANGLES: ";
							Vector3 euler; std::cin >> euler;
							std::cout << "ENTER SCALE: ";
							Vector3 scale; std::cin >> scale;
							std::string lineTillEnd;
							std::getline(std::cin, lineTillEnd);
							Transform trans(pos, euler, scale);
							Transform normalTrans(Vector3(0, 0, 0), euler, Vector3(1, 1, 1));
							for (int i = start; i < meshes.size(); i++){
								for (int j = 0; j < meshes[i].vertextCount(); j++)
									meshes[i].vertex(j) >>= trans;
								for (int j = 0; j < meshes[i].normalCount(); j++)
									meshes[i].normal(j) >>= normalTrans;
							}
							break;
						}
						else if (answer == "" || answer == "n" || answer == "N" || answer == "no" || answer == "No" || answer == "NO") break;
						else std::cout << "YOU'RE NOT ANSWERING MY QUESTIONS, ARE YOU?" << std::endl;
					}
				}
				else std::cout << "UNABLE TO READ THE FILE...." << std::endl;
			}
			int totalVerts = 0, totalNorms = 0, totalTexts = 0, totalFaces = 0, totalTriangles = 0;
			for (int i = 0; i < meshes.size(); i++){
				totalVerts += meshes[i].vertextCount();
				totalNorms += meshes[i].normalCount();
				totalTexts += meshes[i].textureCount();
				totalFaces += meshes[i].faceCount();
				for (int j = 0; j < meshes[i].faceCount(); j++)
					totalTriangles += (meshes[i].indexFace(j).size() - 2);
			}
			std::cout << std::endl << "SCENE TOTAL: " << std::endl;
			std::cout << "   VERTICES:              " << totalVerts << std::endl;
			std::cout << "   VERTEX NORMALS:        " << totalNorms << std::endl;
			std::cout << "   VERTEX TEXTURES:       " << totalTexts << std::endl;
			std::cout << "   FACES:                 " << totalFaces << std::endl;
			std::cout << "   TRIANGLES(from faces): " << totalTriangles << std::endl;
			std::cout << std::endl;
		}

		// ########################################
		// ######## OCTREE CONSTRUCTION: ##########
		// ########################################
		inline static Octree<> constructOctree(const Stacktor<PolyMesh> &meshes, Octree<> *&devOctree){
			Octree<> octree;
			Vertex minVert = meshes[0].vertex(0);
			Vertex maxVert = meshes[0].vertex(0);
			for (int i = 0; i < meshes.size(); i++)
				for (int j = 0; j < meshes[i].vertextCount(); j++){
					if (meshes[i].vertex(j).x < minVert.x) minVert.x = meshes[i].vertex(j).x;
					else if (meshes[i].vertex(j).x > maxVert.x) maxVert.x = meshes[i].vertex(j).x;

					if (meshes[i].vertex(j).y < minVert.y) minVert.y = meshes[i].vertex(j).y;
					else if (meshes[i].vertex(j).y > maxVert.y) maxVert.y = meshes[i].vertex(j).y;

					if (meshes[i].vertex(j).z < minVert.z) minVert.z = meshes[i].vertex(j).z;
					else if (meshes[i].vertex(j).z > maxVert.z) maxVert.z = meshes[i].vertex(j).z;
				}
			octree.reinit(AABB(minVert - EPSILON_VECTOR, maxVert + EPSILON_VECTOR));
			octree.reset();
			for (int i = 0; i < meshes.size(); i++){
				BakedTriMesh mesh;
				meshes[i].bake(mesh);
				/*/
				octree.push(mesh);
				/*/
				octree.put(mesh);
				//*/
				std::cout << "PUSHED " << i << std::endl;
			}
			/*
			octree.build();
			/*/
			octree.reduceNodes();
			//*/
			devOctree = octree.upload();
			if (devOctree != NULL) std::cout << "OCTREE UPLOADED" << std::endl;
			else std::cout << "OCTREE UPLOAD FAILED" << std::endl;
			Octree<> tmpClone = octree;
			return octree;
		}


		class OctreeTestRenderContext{
		private:
			Octree<> octree, *devOctree;

			char windowData[sizeof(Windows::Window)];
			Matrix<Color> image;
			Color *devColor;
			int devColWidth, devColHeight;
			
			bool onDevice, spacePressed;

			POINT cursor;
			Vector3 euler;
			Transform trans;
			bool mouseWasDown;

			struct CPUrenderThread{
				std::thread thread;
				std::condition_variable wait;
				std::condition_variable realease;
				volatile bool b;
			};
			CPUrenderThread *cpuThreads;
			int cpuThreadCount;

			inline Windows::Window& window(){
				return (*((Windows::Window *)windowData));
			}


			// ########################################
			// ############ DEVICE SWITCH: ############
			// ########################################
			inline void switchDevice(){
				if (GetAsyncKeyState(VK_SPACE) & 0x8000){
					if (!spacePressed){
						std::cout << "Changing state...";
						onDevice = !onDevice;
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
			inline void rotate(){
				if (GetKeyState(VK_LBUTTON) & 0x100 && window().inFocus()){
					POINT newCursor; GetCursorPos(&newCursor);
					if (mouseWasDown){
						euler.y += (newCursor.x - cursor.x) / 4.0f;
						euler.x += (newCursor.y - cursor.y) / 4.0f;
						if (euler.x <= -80) euler.x = -80;
						else if (euler.x >= 80) euler.x = 80;
						trans.setEulerAngles(euler);
					}
					else mouseWasDown = true;
					cursor = newCursor;
				}
				else mouseWasDown = false;
			}

			// ########################################
			// ######## DEVICE RENDER ROUTINE: ########
			// ########################################
			inline bool renderOnDevice(int width, int height, int frame){
				if (devColWidth != width || devColHeight != height || devColor == NULL){
					if (devColor != NULL) cudaFree(devColor);
					if (cudaMalloc(&devColor, sizeof(Color) * max(1, width * height)) != cudaSuccess){
						std::cout << "CUDA_MALLOC PROBLEM" << std::endl;
						return false;
					}
					devColWidth = width;
					devColHeight = height;
				}
				cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess){ std::cout << "STREAM ALLOCATION ERROR" << std::endl; return false; }
				color<<<numBlocks(devColWidth, devColHeight), numThreads(), 0, stream>>>(devOctree, devColor, trans, devColWidth, devColHeight, frame);
				bool success = (cudaStreamSynchronize(stream) == cudaSuccess);
				if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
				if (!success){ std::cout << "STREAM JOIN ERROR" << std::endl; return false; }
				window().updateFrameDevice(devColor, width, height);
				return true;
			}

			// ########################################
			// ######### HOST RENDER ROUTINE: #########
			// ########################################
			inline static void cpuRenderThread(const Octree<> *octree, Matrix<Color> *image, Transform trans, int step, int startI, int frame){
				for (int i = startI; i < image->height(); i += step)
					for (int j = 0; j < image->width(); j++)
						colorPixel(*octree, image->operator[](i)[j], trans, i, j, image->width(), image->height(), frame);
			}
			inline void renderOnCPU(int width, int height, int frame){
				if (width != image.width() || height != image.height()) image.setDimensions(width, height);
				int numThreads = min(max(std::thread::hardware_concurrency(), 1), 32);
				std::thread threads[32];
				for (int i = 0; i < numThreads; i++) threads[i] = std::thread(cpuRenderThread, &octree, &image, trans, numThreads, i, frame);
				for (int i = 0; i < numThreads; i++) threads[i].join();
				window().updateFrameHost(image);
			}

			// ########################################
			// ############# FRAME RENDER: ############
			// ########################################
			inline bool render(int frame){
				int width, height; 
				window().getDimensions(width, height);
				if (onDevice) return renderOnDevice(width, height, frame);
				else renderOnCPU(width, height, frame);
				return true;
 			}

			// ########################################
			// ############### FPS DUMP: ##############
			// ########################################
			inline static void dumpFPS(const int frame, long &time){
				if (frame % 128 == 0 && frame > 0){
					long newTime = clock();
					long deltaTime = (newTime - time);
					float avgDeltaTime = ((float)deltaTime) / 128.0f;
					std::cout << "CLOCK: " << avgDeltaTime << " (" << CLOCKS_PER_SEC / avgDeltaTime << " fps)" << std::endl;
					time = newTime;
				}
			}

		public:
			// ########################################
			// ######## READING & PREPARATION: ########
			// ########################################
			inline OctreeTestRenderContext(){
				// ################################
				// ############# INTRO: ###########
				std::cout << std::fixed << std::setprecision(2);
				dumpConfiguration();
				// ################################
				// ############# DATA: ############
				Stacktor<PolyMesh> meshes; readMeshes(meshes);
				octree = constructOctree(meshes, devOctree);
				// ############ WINDOW: ###########
				devColor = NULL;
				devColWidth = 0;
				devColHeight = 0;
				// ############ RENDER: ###########
				std::cout << "READY TO RENDER" << std::endl;
				onDevice = true;
				spacePressed = false;
				// ########### ROTATION: ##########
				euler(0, 0, 0);
				mouseWasDown = true;
				// ######### CPU THREADS: #########
				Octree<Vertex> octo;
				octo.put(Vertex(0, 0, 0));
				octo.cast(Ray(Vector3(-32, -32, -32), Vector3(1, 1, 1)));
			}

			// ########################################
			// ################ CLEANUP: ##############
			// ########################################
			inline ~OctreeTestRenderContext(){
				if (devOctree != NULL){
					if (Octree<>::dispose(devOctree)) std::cout << "DEVICE OCTREE DIPOSED SUCCESSFULY" << std::endl;
					else std::cout << "ERROR DISPOSING OF DEVICE OCTREE" << std::endl;
					cudaFree(devOctree);
				}
				if (devColor != NULL) cudaFree(devColor);
			}

			// ########################################
			// ############# RENDER TEST: #############
			// ########################################
			inline void runTest(){
				new(&window()) Windows::Window("OCTREE TEST WINDOW");
				int frame = 0; long time = clock();
				while (true){
					if (window().dead()) break;
					switchDevice();
					rotate();
					if (!render(frame)) continue;
					dumpFPS(frame, time);
					frame++;
				}
				window().~Window();
			}
		};

		// ########################################
		// ########## BASIC OCTREE TEST: ##########
		// ########################################
		inline static void test(){
			OctreeTestRenderContext context;
			context.runTest();
		}
	}

	/*
	Tests basic capabilities of Octree by rendering normals with backward ray tracing
	*/
	inline static void test(){
		Tests::runTest(Private::test, "Testing Octree");
	}

}

