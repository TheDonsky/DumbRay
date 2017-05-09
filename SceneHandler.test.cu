#include "SceneHandler.test.cuh"
#include "SceneHandler.cuh"
#include "MeshReader.test.h"
#include "SimpleDirectionalLight.cuh"
#include "DefaultPerspectiveLense.cuh"
#include "Tests.h"
#include <iostream>
#include <thread>
#include <mutex>
#include "Semaphore.h"



namespace SceneHandlerTest {
	namespace Private{
		__global__ void kernel(const Scene<BakedTriFace> *scene) {
			Vector2 screenPoint = ((Vector2(threadIdx.x, blockIdx.x) / Vector2(blockDim.x, gridDim.x)) - Vector2(0.5f, 0.0f));
			scene->geometry.cast(scene->cameras[0].lense.getScreenPhoton(screenPoint).ray);
		}

		static void runKernels(const volatile bool *quit, const SceneHandler<BakedTriFace> *scene, int index, std::mutex *ioLock, Semaphore *initSem, std::mutex *exitLock) {
			if (!scene->selectGPU(index)) {
				ioLock->lock();
				std::cout << "Thread " << index << " could not select the GPU." << std::endl;
				ioLock->unlock();
				initSem->post();
				return;
			}
			if (scene->getHandleGPU(index) == NULL) {
				ioLock->lock();
				std::cout << "GPU handle missing for thread " << index << "." << std::endl;
				ioLock->unlock();
				initSem->post();
				return;
			}
			ioLock->lock();
			std::cout << "Thread " << index << " active." << std::endl;
			ioLock->unlock();
			initSem->post();
			while (true) {
				if (*quit) break;
				kernel<<<256,256>>>(scene->getHandleGPU(index));
				if (cudaDeviceSynchronize() != cudaSuccess) {
					exitLock->lock();
					ioLock->lock();
					std::cout << "Thread " << index << " failed." << std::endl;
					ioLock->unlock();
					exitLock->unlock();
					return;
				}
			}
			ioLock->lock();
			std::cout << "Thread " << index << " quit." << std::endl;
			ioLock->unlock();
		}
		static void makeScene(Scene<BakedTriFace> &scene) {
			Stacktor<PolyMesh> meshes; 
			MeshReaderTest::readMeshes(meshes);
			for (int i = 0; i < meshes.size(); i++) {
				scene.geometry.push(meshes[i].bake());
				std::cout << "\rPUSHED " << i;
			}
			std::cout << std::endl << "BUILDING. PLEASE WAIT..." << std::endl;
			scene.geometry.build();
			scene.lights.flush(1);
			Vector3 direction = Vector3(0.2f, -0.4f, 0.7f).normalized();
			scene.lights[0].use<SimpleDirectionalLight>(Photon(Ray(-direction * 10000.0f, direction), Color(1.0f, 1.0f, 1.0f)));
			scene.cameras.flush(1);
			scene.cameras[0].transform.setPosition(Vector3(0, 0, -128));
			scene.cameras[0].lense.use<DefaultPerspectiveLense>(60.0f);
		}
		static void runTest() {
			Scene<BakedTriFace> scene;
			makeScene(scene);
			SceneHandler<BakedTriFace> handler(scene);
			std::cout << "Uploading..." << std::endl;
			handler.uploadToEveryGPU();
			volatile bool quit = false;
			std::mutex ioLock;
			Semaphore initSem;
			std::mutex exitLock;
			exitLock.lock();
			std::thread *threads = new std::thread[handler.gpuCount()];
			for (int i = 0; i < handler.gpuCount(); i++)
				threads[i] = std::thread(runKernels, &quit, &handler, i, &ioLock, &initSem, &exitLock);
			for (int i = 0; i < handler.gpuCount(); i++) initSem.wait();
			ioLock.lock();
			std::cout << "Should likely be running some kernels on GPU-s. Enter anthing to quit... ";
			ioLock.unlock();
			std::string s;
			std::getline(std::cin, s);
			quit = true;
			exitLock.unlock();
			for (int i = 0; i < handler.gpuCount(); i++) threads[i].join();
			delete[] threads;
		}
		static void test() {
			while (true) {
				std::cout << "Enter anthing to run SceneHandler test: ";
				std::string s;
				std::getline(std::cin, s);
				if (s.length() <= 0) break;
				runTest();
			}
		}
	}

	void test() {
		Tests::runTest(Private::test, "Testing SceneHandler");
		cudaSetDevice(0);
	}
}



