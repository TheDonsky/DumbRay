#include "DumbRenderer.test.cuh"
#include "DumbRenderer.cuh"
#include "../BufferedRenderProcess/BufferedRenderProcess.test.cuh"
#include "../../Screen/FrameBuffer/BlockBasedFrameBuffer/BlockBasedFrameBuffer.cuh"
#include "../../Objects/Components/Shaders/DefaultShader/DefaultShader.cuh"
#include "../../Objects/Components/Shaders/SimpleStochasticShader/SimpleStochasticShader.cuh"
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

			template<typename ShaderType, typename LightType, typename LenseType>
			void init() {
				scene.materials.cpuHandle()->flush(1);
				scene.materials.cpuHandle()->top().use<ShaderType>();
				scene.lights.cpuHandle()->flush(3);
				scene.lights.cpuHandle()->operator[](0).use<LightType>(
					Color(0.125f, 0.125f, 0.5f, 1.0f), Vector3(1, -1, 0), 512.0f);
				scene.lights.cpuHandle()->operator[](1).use<LightType>(
					Color(0.125f, 0.5f, 0.125f, 1.0f), Vector3(-0.5f, -1, 0.580611f), 512.0f);
				scene.lights.cpuHandle()->operator[](2).use<LightType>(
					Color(0.5f, 0.125f, 0.125f, 1.0f), Vector3(-0.5f, -1, -0.580611f), 512.0f);
				camera.cpuHandle()->lense.use<LenseType>();
				camera.cpuHandle()->transform = Transform(
					Vertex(0.0f, 0.0f, 0.0f),
					Vector3(48.0f, 32.0f, 0.0f),
					Vector3(1.0f, 1.0f, 1.0f));
				camera.cpuHandle()->transform.setPosition(
					camera.cpuHandle()->transform.back() * 128.0f);
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
		};

		inline BufferedRenderer *makeRenderer(
			const Renderer::ThreadConfiguration &configuration, void *contextAddr) {
			volatile Context *context = ((volatile Context*)contextAddr);
			DumbRenderer *renderer = new DumbRenderer(configuration);
			renderer->setScene(((DumbRenderer::SceneType*)&context->scene));
			renderer->setCamera(((DumbRenderer::CameraManager*)&context->camera));
			return renderer;
		}

		template<typename ShaderType, typename LightType, typename LenseType>
		inline void simpleTestCase() {
			volatile Context *context = new Context();
			((Context*)context)->init<ShaderType, LightType, LenseType>();
			FrameBufferManager bufferA, bufferB;
			bufferA.cpuHandle()->use<BlockBuffer>();
			bufferB.cpuHandle()->use<BlockBuffer>();
			BufferedRenderProcessTest::runTestGauntlet(makeRenderer, ((void*)context),
				&bufferA, &bufferB, BufferedRenderProcessTest::TEST_RUN_FULL_GAUNTLET);
			delete context;
		}
	}

	void simpleNonInteractiveTest() {
		simpleTestCase<DefaultShader, SimpleDirectionalLight, DefaultPerspectiveLense>();
	}


	void simpleNonInteractiveStochsticTest() {
		simpleTestCase<SimpleStochasticShader, SimpleSoftDirectionalLight, SimpleStochasticLense>();
	}
}
