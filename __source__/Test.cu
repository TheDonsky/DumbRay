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
__global__ void testSceneHandle(Scene<BakedTriFace> *scene) {
	RaycastHit<Shaded<BakedTriFace> > hit;
	if (scene->geometry.cast(Ray(Vertex::zero(), Vector3::one()), hit))
		printf("Raycast hit something\n");
	printf("Raycast hit nothing\n");
	PhotonPack result;
	bool noShadows;
	printf("CALLING scene->lights[0].getPhoton()...\n");
	scene->lights[0].getPhoton(Vertex::zero(), &noShadows, result);
	printf("CALL FOR scene->lights[0].getPhoton() JUST ENDED\n");
	printf("len(illuminationPhotons): %d\n", result.size());
	result.clear();
	scene->cameras[0].getPhoton(Vector2::zero(), result);
	printf("len(screenPhotons): %d\n", result.size());
	/*
	LenseFunctionPack functions;
	functions.use<DefaultPerspectiveLense>();
	//*/
}
bool sanityCheck() {
	if (cudaSetDevice(0) != cudaSuccess) {
		std::cout << "SETTING DEVICE FAILED QUITE A BIT MISERABLY" << std::endl;
		return false;
	}
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
	sceneHandler.uploadToEveryGPU();
	
	testSceneHandle<<<1, 1>>>(sceneHandler.getHandleGPU(0));
	bool rv;
	if (cudaDeviceSynchronize() != cudaSuccess) {
		std::cout << "DEVICE FAILED..." << std::endl;
		rv = false;
	}
	else rv = true;
	
	std::string line;
	std::cout << "PRESS ENTER TO CONTINUE...";
	std::getline(std::cin, line);
	std::cout << std::endl << std::endl << std::endl;
	return rv;
}
void testBackwardRenderer() {
	if (!sanityCheck()) return;
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
				Renderer::ThreadConfiguration());
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
				if (i % 4 == 0) {
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
int main(){
	testBackwardRenderer();
	test();
	return 0;
}
