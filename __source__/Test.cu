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
#include"DataStructures/Objects/Scene/Raycasters/ShadedOctree/ShadedOctree.cuh"
struct MockRaycaster {
	ShadedOctree<BakedTriFace> *object;

	typedef bool(*CastBreaker)(RaycastHit<Shaded<BakedTriFace> > &hit, const Ray &ray, bool &rv);
	__dumb__ bool cast(const Ray &r, RaycastHit<Shaded<BakedTriFace> > &out, bool ignored, CastBreaker castBreaker)const {
		//return true;
		return object->cast(r, out, ignored, castBreaker);
	}
}; 
__global__ void mdaa(ShadedOctree<BakedTriFace> *object, mutable RaycastFunctionPack<Shaded<BakedTriFace> > *functionPack) {
	//functionPack->use<ShadedOctree<BakedTriFace> >();
	RaycastFunctionPack<Shaded<BakedTriFace> > fn;
	fn.use<ShadedOctree<BakedTriFace> >();

	(*functionPack).use<MockRaycaster>();
	RaycastHit<Shaded<BakedTriFace> > hit;
	functionPack->cast(object, Ray(Vector3(1, 1, 1), Vector3(1, 1 ,1)), hit, false, NULL);
	fn.cast(object, Ray(Vector3(1, 1, 1), Vector3(1, 1, 1)), hit, false, NULL);
	printf("HERE I AM...\n");
}
typedef Octree<BakedTriFace> OctreeType;
int main(){
	ShadedOctree<BakedTriFace> object;
	RaycastFunctionPack<Shaded<BakedTriFace> > *fn;
	cudaMalloc(&fn, sizeof(RaycastFunctionPack<Shaded<BakedTriFace> >));
	mdaa << <1, 1 >> > (object.upload(), fn);
	//Generic<RaycastFunctionPack<BakedTriFace> > raycaster;
	//raycaster.use<OctreeType>();
	cudaDeviceSynchronize();
	test();
}
