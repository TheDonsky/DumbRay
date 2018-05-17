#include "DumbRenderer.test.cuh"
#include "DumbRenderer.cuh"
#include "../BufferedRenderProcess/BufferedRenderProcess.test.cuh"
#include "../../Screen/FrameBuffer/BlockBasedFrameBuffer/BlockBasedFrameBuffer.cuh"
#include "../../Objects/Components/Shaders/DefaultShader/DefaultShader.cuh"
#include "../../Objects/Components/Shaders/SimpleStochasticShader/SimpleStochasticShader.cuh"
#include "../../Objects/Components/Shaders/DumbBasedShader/DumbBasedShader.cuh"
#include "../../Objects/Components/Lenses/DefaultPerspectiveLense/DefaultPerspectiveLense.cuh"
#include "../../Objects/Components/Lenses/SimpleStochasticLense/SimpleStochasticLense.cuh"
#include "../../Objects/Scene/Lights/SimpleDirectionalLight/SimpleDirectionalLight.cuh"
#include "../../Objects/Scene/Lights/SimpleSoftDirectionalLight/SimpleSoftDirectionalLight.cuh"
#include "../../../Namespaces/MeshReader/MeshReader.test.h"
#include "../../Objects/Scene/Raycasters/ShadedOctree/ShadedOctree.cuh"

namespace DumbRendererTest {
	namespace {
		struct Context {
			DumbRenderer::SceneType scene;
			DumbRenderer::CameraManager camera;

			template<typename LightType>
			static void addTripleLights(Context *self) {
				int start = self->scene.lights.cpuHandle()->size();
				self->scene.lights.cpuHandle()->flush(3);
				self->scene.lights.cpuHandle()->operator[](start).use<LightType>(
					Color(0.125f, 0.125f, 0.5f, 1.0f), Vector3(1, -1, 0), 512.0f);
				self->scene.lights.cpuHandle()->operator[](start + 1).use<LightType>(
					Color(0.125f, 0.5f, 0.125f, 1.0f), Vector3(-0.5f, -1, 0.580611f), 512.0f);
				self->scene.lights.cpuHandle()->operator[](start + 2).use<LightType>(
					Color(0.5f, 0.125f, 0.125f, 1.0f), Vector3(-0.5f, -1, -0.580611f), 512.0f);
			}

			template<typename LightType>
			static void addSingleLight(Context *self) {
				int start = self->scene.lights.cpuHandle()->size();
				self->scene.lights.cpuHandle()->flush(1);
				self->scene.lights.cpuHandle()->operator[](start).use<LightType>(
					Color(4.0f, 4.0f, 4.0f, 4.0f), Vector3(0.25f, -1.0f, -0.25f).normalized(), 512.0f);
			}

			void addObjects() {
				Stacktor<PolyMesh> meshes;
				MeshReaderTest::readMeshes(meshes);
				for (int i = 0; i < meshes.size(); i++) {
					BakedTriMesh bakedMesh = meshes[i].bake();
					for (int j = 0; j < bakedMesh.size(); j++)
						scene.geometry.cpuHandle()->push(
							DumbRenderer::SceneType::GeometryUnit(bakedMesh[j], 0));
				}
				scene.geometry.cpuHandle()->build();
			}

			template<typename ShaderType, typename LenseType>
			void init(void(*addLightsFn)(Context*), const ShaderType &shader) {
				scene.materials.cpuHandle()->flush(1);
				scene.materials.cpuHandle()->top().use<ShaderType>(shader);
				addLightsFn(this);
				camera.cpuHandle()->lense.use<LenseType>();
				camera.cpuHandle()->transform = Transform(
					Vertex(0.0f, 0.0f, 0.0f),
					Vector3(48.0f, 32.0f, 0.0f),
					Vector3(1.0f, 1.0f, 1.0f));
				camera.cpuHandle()->transform.setPosition(
					camera.cpuHandle()->transform.back() * 128.0f);
				
				addObjects();
			}
		};

		inline BufferedRenderer *makeRenderer(
			const Renderer::ThreadConfiguration &configuration, void *contextAddr) {
			volatile Context *context = ((volatile Context*)contextAddr);
			DumbRenderer *renderer = new DumbRenderer(configuration);
			renderer->setScene(((DumbRenderer::SceneType*)&context->scene));
			renderer->setCamera(((DumbRenderer::CameraManager*)&context->camera));
			return renderer;
		}

		template<typename ShaderType, typename LenseType>
		inline void simpleTestCase(void(*addLightsFn)(Context*), const ShaderType &shader, uint32_t tests) {
			volatile Context *context = new Context();
			((Context*)context)->init<ShaderType, LenseType>(addLightsFn, shader);
			FrameBufferManager bufferA, bufferB;
			bufferA.cpuHandle()->use<BlockBuffer>();
			bufferB.cpuHandle()->use<BlockBuffer>();
			BufferedRenderProcessTest::runTestGauntlet(makeRenderer, ((void*)context),
				&bufferA, &bufferB, tests);
			delete context;
		}
	}

	void simpleNonInteractiveTestFull() {
		simpleTestCase<DefaultShader, DefaultPerspectiveLense>(
			Context::addTripleLights<SimpleDirectionalLight>, DefaultShader(), BufferedRenderProcessTest::TEST_RUN_FULL_GAUNTLET);
	}

	void simpleNonInteractiveStochsticTestFull() {
		simpleTestCase<SimpleStochasticShader, SimpleStochasticLense>(
			Context::addTripleLights<SimpleSoftDirectionalLight>, SimpleStochasticShader(), BufferedRenderProcessTest::TEST_RUN_FULL_GAUNTLET);
	}

	void testDumbBasedGoldFull() {
		simpleTestCase<DumbBasedShader, SimpleStochasticLense>(
			Context::addSingleLight<SimpleSoftDirectionalLight>, DumbBasedShader::roughGold(), BufferedRenderProcessTest::TEST_RUN_FULL_GAUNTLET);
	}
	void testDumbBasedGlossyFull() {
		simpleTestCase<DumbBasedShader, SimpleStochasticLense>(
			Context::addSingleLight<SimpleSoftDirectionalLight>, DumbBasedShader::glossyFinish(), BufferedRenderProcessTest::TEST_RUN_FULL_GAUNTLET);
	}
	void testDumbBasedMatteFull() {
		simpleTestCase<DumbBasedShader, SimpleStochasticLense>(
			Context::addSingleLight<SimpleSoftDirectionalLight>, DumbBasedShader::matteFinish(), BufferedRenderProcessTest::TEST_RUN_FULL_GAUNTLET);
	}


	void simpleNonInteractiveTest() {
		simpleTestCase<DefaultShader, DefaultPerspectiveLense>(
			Context::addTripleLights<SimpleDirectionalLight>, DefaultShader(), BufferedRenderProcessTest::TEST_MULTI_ITER_CPU_AND_GPU);
	}

	void simpleNonInteractiveStochsticTest() {
		simpleTestCase<SimpleStochasticShader, SimpleStochasticLense>(
			Context::addTripleLights<SimpleSoftDirectionalLight>, SimpleStochasticShader(), BufferedRenderProcessTest::TEST_MULTI_ITER_CPU_AND_GPU);
	}

	void testDumbBasedGold() {
		simpleTestCase<DumbBasedShader, SimpleStochasticLense>(
			Context::addSingleLight<SimpleSoftDirectionalLight>, DumbBasedShader::roughGold(), BufferedRenderProcessTest::TEST_MULTI_ITER_CPU_AND_GPU);
	}
	void testDumbBasedGlossy() {
		simpleTestCase<DumbBasedShader, SimpleStochasticLense>(
			Context::addSingleLight<SimpleSoftDirectionalLight>, DumbBasedShader::glossyFinish(), BufferedRenderProcessTest::TEST_MULTI_ITER_CPU_AND_GPU);
	}
	void testDumbBasedMatte() {
		simpleTestCase<DumbBasedShader, SimpleStochasticLense>(
			Context::addSingleLight<SimpleSoftDirectionalLight>, DumbBasedShader::matteFinish(), BufferedRenderProcessTest::TEST_MULTI_ITER_CPU_AND_GPU);
	}
}
