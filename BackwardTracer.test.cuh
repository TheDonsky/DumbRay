#pragma once
#include "BackwardTracer.cuh"
#include "MeshReader.test.h"
#include "DefaultPerspectiveLense.cuh"
#include "SimpleDirectionalLight.cuh"
#include "Windows.h"
#include "Tests.h"




namespace BackwardTracerTest {
	namespace Private {


		inline static bool initializeImage(Handler<Matrix<Color> > &image) {
			if (!image.createHandle()) {
				std::cout << "ERROR: Unable to create host image" << std::endl;
				return false;
			}
			else if (!image.uploadHostHandleToDevice()) {
				std::cout << "ERROR: Unable to create GPU image" << std::endl;
				cudaDeviceReset();
				return false;
			}
			else return true;
		}
		inline static bool swapState(bool &usingDevice, DumbBackTracer &renderer, bool &shouldSwapState) {
			if (shouldSwapState) {
				if ((!renderer.useHost(usingDevice)) || (!renderer.useDevice(!usingDevice))) {
					if (usingDevice) std::cout << "ERROR: Unable to use host" << std::endl;
					else std::cout << "ERROR: Unable to use device" << std::endl;
					cudaDeviceReset();
					return false;
				}
				else {
					usingDevice = (!usingDevice);
					if (usingDevice) std::cout << "USING DEVICE" << std::endl;
					else std::cout << "USING HOST" << std::endl;
					shouldSwapState = false;
					return true;
				}
			}
			else return true;
		}
		inline static bool cleanFrame(DumbBackTracer &renderer, int frame) {
			if (!renderer.cleanImage(ColorRGB((frame % 256) / 256.0f, ((frame / 256) % 256) / 256.0f, ((frame / 256 / 256) % 256) / 256.0f))) {
				std::cout << "ERROR: Unable to clean image" << std::endl;
				cudaDeviceReset();
				return false;
			}
			else return true;
		}
		inline static bool iterate(DumbBackTracer &renderer) {
			if (!renderer.iterate()) {
				std::cout << "ERROR: Unable to render image..." << std::endl;
				cudaDeviceReset();
				return false;
			}
			else return true;
		}
		inline static bool resetResolution(DumbBackTracer &renderer, Handler<Matrix<Color> > &image, const Windows::Window &window) {
			int width, height;
			if (!window.getDimensions(width, height)) {
				std::cout << "ERROR: Unable to get dimensions" << std::endl;
				cudaDeviceReset();
				return false;
			}
			else if (image.hostHandle->width() != width || image.hostHandle->height() != height) {
				image.hostHandle->setDimensions(width, height);
				if (!image.uploadHostHandleToDevice(true)) {
					std::cout << "ERROR: Unable to set image resolution" << std::endl;
					cudaDeviceReset();
					return false;
				}
				else if (!renderer.setResolution(width, height)) {
					std::cout << "ERROR: Unable to set resolution" << std::endl;
					cudaDeviceReset();
					return false;
				}
			}
			return true;
		}
		inline static bool displayImage(DumbBackTracer &renderer, Handler<Matrix<Color> > &image, Windows::Window &window, bool usingDevice) {
			if (!renderer.loadOutput(image)) {
				std::cout << "Unable to load image" << std::endl;
				cudaDeviceReset();
				return false;
			}
			else if(!window.dead()) {
				if (usingDevice)
					window.updateFrameDevice(image.deviceHandle);
				else window.updateFrameHost(*image.hostHandle);
			}
			return true;
		}
		inline static void disposeImage(Handler<Matrix<Color> > &image) {
			if (!image.destroyHandles()) {
				std::cout << "ERROR: Unable to dispose of image handles" << std::endl;
				cudaDeviceReset();
			}
		}

		inline static bool rotateCamera(Windows::Window &window, bool &mouseWasDown, POINT &cursor, Handler<Camera> &camera){
			bool altered = false;
			if (window.inFocus()) {
				if (GetKeyState(VK_LBUTTON) & 0x100) {
					POINT newCursor; GetCursorPos(&newCursor);
					if (mouseWasDown) {
						Vector3 euler = camera.object().transform.getEulerAngles();
						euler.y += (newCursor.x - cursor.x) / 4.0f;
						euler.x += (newCursor.y - cursor.y) / 4.0f;
						if (euler.x <= -80) euler.x = -80;
						else if (euler.x >= 80) euler.x = 80;
						camera.object().transform.setEulerAngles(euler);
						altered = true;
					}
					else mouseWasDown = true;
					cursor = newCursor;
				}
				else mouseWasDown = false;
			}
			else mouseWasDown = false;
			if (altered) {
				camera.object().transform.setPosition(camera.object().transform.back() * 128.0f);
				if (!camera.uploadHostHandleToDevice(true)) {
					std::cout << "ERROR refreshing camera..." << std::endl;
					cudaDeviceReset();
					return false;
				}
			}
			return true;
		}

		inline static void windowLoop(DumbBackTracer &renderer, Handler<Camera> &camera) {
			Windows::Window window;
			Handler<Matrix<Color> > image;
			if (!initializeImage(image)) return;
			bool shouldSwapState = true;
			bool usingDevice = false;
			int frame = 0;
			long long t = clock();
			bool mouseWasDown = false;
			POINT cursor;
			if (cleanFrame(renderer, 0))
				while (!window.dead()) {
					if (!rotateCamera(window, mouseWasDown, cursor, camera));
					if (!swapState(usingDevice, renderer, shouldSwapState)) break;
					if (!resetResolution(renderer, image, window)) break;
					/*
					if (!cleanFrame(renderer, frame)) break;
					/*/
					renderer.resetIterations();
					//*/
					if (!iterate(renderer)) break;
					if (!displayImage(renderer, image, window, usingDevice)) break;
					frame++;
					if (frame % 64 == 0) {
						float avgClock = (((float)(clock() - t)) / 64.0f);
						float time = avgClock / CLOCKS_PER_SEC;
						std::cout << "Fps: " << (1.0f / time) << " (Average: frame time: " << (time) << "; clock: " << avgClock << ")" << std::endl;
						t = clock();
					}
				}
			disposeImage(image);
		}

		inline static void test() {
			DumbBackTracer renderer;
			Stacktor<PolyMesh> meshes; MeshReaderTest::readMeshes(meshes);
			ShadedOctree<BakedTriFace> scene;
			for (int i = 0; i < meshes.size(); i++)
				scene.push(meshes[i].bake());
			scene.build();
			Stacktor<Light> lights;
			lights.flush(1);
			Vector3 direction = Vector3(0.2, -0.4, 0.7).normalized();
			lights[0].use<SimpleDirectionalLight>(Photon(Ray(-direction * 10000.0f, direction), Color(1.0f, 1.0f, 1.0f)));
			Camera camera;
			camera.transform.setPosition(Vector3(0, 0, -128));
			camera.lense.use<DefaultPerspectiveLense>(60.0f);

			Handler<Camera> cameraHandler;
			cameraHandler.hostHandle = &camera;
			if (!cameraHandler.uploadHostHandleToDevice()) {
				std::cout << "ERROR UPLOADING CAMERA..." << std::endl;
				cudaDeviceReset();
				return;
			}//*/
			Handler<ShadedOctree<BakedTriFace> > sceneHandler;
			sceneHandler.hostHandle = &scene;
			if (!sceneHandler.uploadHostHandleToDevice()) {
				std::cout << "ERROR UPLOADING SCENE..." << std::endl;
				cudaDeviceReset();
				return;
			}//*/
			Handler<Stacktor<Light> > lightsHandler;
			lightsHandler.hostHandle = &lights;
			if (!lightsHandler.uploadHostHandleToDevice()) {
				std::cout << "ERROR UPLOADING LIGHTS..." << std::endl;
				cudaDeviceReset();
				return;
			}//*/


			Handler<const Camera> cameraHandlerConst;
			cameraHandlerConst.hostHandle = cameraHandler.hostHandle;
			cameraHandlerConst.deviceHandle = cameraHandler.deviceHandle;
			Handler<const ShadedOctree<BakedTriFace> > sceneHandlerConst;
			sceneHandlerConst.hostHandle = sceneHandler.hostHandle;
			sceneHandlerConst.deviceHandle = sceneHandler.deviceHandle;
			Handler<const Stacktor<Light> > lightsHandlerConst;
			lightsHandlerConst.hostHandle = lightsHandler.hostHandle;
			lightsHandlerConst.deviceHandle = lightsHandler.deviceHandle;

			renderer.setCamera(cameraHandlerConst);
			renderer.setScene(sceneHandlerConst, lightsHandlerConst);

			windowLoop(renderer, cameraHandler);
			//*
			if ((!cameraHandler.destroyDeviceHandle()) || (!sceneHandler.destroyDeviceHandle()) || (!lightsHandler.destroyDeviceHandle())) {
				std::cout << "Unable to dispose device handle..." << std::endl;
				cudaDeviceReset();
				return;
			}//*/
		}
	}

	inline static void test() {
		while (true) {
			Tests::runTest(Private::test, "Testing BackwardTracer");
			std::cout << "Enter anything to re-run the test: ";
			std::string s;
			std::getline(std::cin, s);
			if (s.length() <= 0) break;
		}
	}
}

