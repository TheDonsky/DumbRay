#include"DataStructures/GeneralPurpose/Stacktor/Stacktor.test.cuh"
#include"DataStructures/GeneralPurpose/IntMap/IntMap.test.h"
#include"DataStructures/GeneralPurpose/Cutex/Cutex.test.cuh"
#include"DataStructures/Objects/Scene/Raycasters/Octree/Octree.test.cuh"
#include"DataStructures/GeneralPurpose/Generic/Generic.test.cuh"
#include"DataStructures/Objects/Components/Lenses/Lense.test.cuh"
#include"DataStructures/GeneralPurpose/Handler/Handler.test.cuh"
#include"DataStructures/Objects/Components/Shaders/Material.test.cuh"
#include"DataStructures/GeneralPurpose/TypeTools/TypeTools.test.cuh"
#include"DataStructures/Objects/Scene/SceneHandler/SceneHandler.test.cuh"
#include"DataStructures/Renderers/Renderer/Renderer.test.cuh"
#include"DataStructures/Screen/FrameBuffer/MemoryMappedFrameBuffer/MemoryMappedFrameBuffer.test.cuh"
#include"DataStructures/Screen/FrameBuffer/BlockBasedFrameBuffer/BlockBasedFrameBuffer.test.cuh"
#include"DataStructures/Screen/FrameBuffer/FrameBuffer.test.cuh"
#include"DataStructures/Renderers/DumbRenderer/DumbRenderer.test.cuh"
#include"DataStructures/GeneralPurpose/DumbRand/DumbRand.test.cuh"
#include"Namespaces/Images/Images.test.cuh"
#include"Namespaces/Device/Device.cuh"
#include"Namespaces/Dson/Dson.test.h"
#include"Playground/Checkerboard/Checkerboard.cuh"
#include <map>
#include <string>
#include <ctype.h>
#include <iomanip>


namespace {
	typedef void(*TestFunction)();
	struct TestEntry {
		std::string description;
		TestFunction function;
	};
	static void test() {
		std::cout << std::fixed << std::setprecision(4);
		cudaSetDevice(0);
		Device::dumpCurrentDevice();
		std::map<std::string, TestEntry> tests;
		tests["scene_handler"] = { "Basic test for SceneHandler structure", SceneHandlerTest::test };
		tests["renderer"] = { "Basic test for standrad Renderer pipeline", RendererTest::test };
		tests["type_tools"] = { "General tests for TypeTools and it's default implementations", TypeToolsTest::test };
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
		tests["block_frame_buffer"] = { "Basic test for BlockBasedFrameBuffer", BlockBasedFrameBufferTest::test };
		
		tests["dumb_renderer_simple_full"] = { "Basic performance test for DumbRenderer (full gauntlet)", DumbRendererTest::simpleNonInteractiveTestFull };
		tests["dumb_renderer_stochastic_full"] = { "Basic performance test with stochastic entities for DumbRenderer (full gauntlet)", DumbRendererTest::simpleNonInteractiveStochsticTestFull };
		tests["dumb_based_gold_full"] = { "A simple general test for dumb renderer with DumbBasedShader::roughGold (full gauntlet)", DumbRendererTest::testDumbBasedGoldFull };
		tests["dumb_based_gold_dim_full"] = { "A simple general test for dumb renderer with DumbBasedShader::roughGold (dim light) (full gauntlet)", DumbRendererTest::testDumbBasedGoldDimFull };
		tests["dumb_based_gold_glossy_full"] = { "A simple general test for dumb renderer with DumbBasedShader::glossyGold (full gauntlet)", DumbRendererTest::testDumbBasedGoldGlossyFull };
		tests["dumb_based_glossy_full"] = { "A simple general test for dumb renderer with DumbBasedShader::glossyFinish (full gauntlet)", DumbRendererTest::testDumbBasedGlossyFull };
		tests["dumb_based_matte_full"] = { "A simple general test for dumb renderer with DumbBasedShader::matteFinish (full gauntlet)", DumbRendererTest::testDumbBasedMatteFull };

		tests["dumb_renderer_simple"] = { "Basic performance test for DumbRenderer", DumbRendererTest::simpleNonInteractiveTest };
		tests["dumb_renderer_stochastic"] = { "Basic performance test with stochastic entities for DumbRenderer", DumbRendererTest::simpleNonInteractiveStochsticTest };
		tests["dumb_based_gold"] = { "A simple general test for dumb renderer with DumbBasedShader::roughGold", DumbRendererTest::testDumbBasedGold };
		tests["dumb_based_gold_dim"] = { "A simple general test for dumb renderer with DumbBasedShader::roughGold (dim light)", DumbRendererTest::testDumbBasedGoldDim };
		tests["dumb_based_gold_glossy"] = { "A simple general test for dumb renderer with DumbBasedShader::glossyGold", DumbRendererTest::testDumbBasedGoldGlossy };
		tests["dumb_based_glossy"] = { "A simple general test for dumb renderer with DumbBasedShader::glossyFinish", DumbRendererTest::testDumbBasedGlossy };
		tests["dumb_based_matte"] = { "A simple general test for dumb renderer with DumbBasedShader::matteFinish", DumbRendererTest::testDumbBasedMatte };
		
		tests["dumb_rand"] = { "Simple tests for DumbRand", DumbRandTest::test };
		tests["save_buffer_png"] = { "Simple tests for saving FrameBuffer as a png file", ImagesTest::testSavePng };
		tests["dson_to_string"] = { "Test of how well Dson gets translated to string", DsonTest::testToString };
		tests["dson_from_string"] = { "Test of how well Dson gets parsed from string", DsonTest::testFromString };
		tests["test_checkerboard"] = { "Playground test of checkerboard rendering", CheckerboardTest::test };
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
				command += (char)tolower(line[i]);
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

namespace TestingModule {
	int run() {
		test();
		return 0;
	}
}
