#include "DumbRenderer.test.cuh"
#include "DumbRenderer.cuh"
#include "../BufferedRenderProcess/BufferedRenderProcess.test.cuh"
#include "../../Screen/FrameBuffer/BlockBasedFrameBuffer/BlockBasedFrameBuffer.cuh"
#include "../../Objects/Components/Shaders/DefaultShader/DefaultShader.cuh"
#include "../../Objects/Components/Lenses/DefaultPerspectiveLense/DefaultPerspectiveLense.cuh"
#include "../../Objects/Scene/Lights/SimpleDirectionalLight/SimpleDirectionalLight.cuh"
#include "../../../Namespaces/MeshReader/MeshReader.test.h"
#include "../../Objects/Scene/Raycasters/ShadedOctree/ShadedOctree.cuh"

namespace DumbRendererTest {
	namespace {
		struct Context {
			DumbRenderer::SceneType scene;
			DumbRenderer::CameraManager camera;

			inline Context() {
				scene.materials.cpuHandle()->flush(1);
				scene.materials.cpuHandle()->top().use<DefaultShader>();
				scene.lights.cpuHandle()->flush(3);
				scene.lights.cpuHandle()->operator[](0).use<SimpleDirectionalLight>(
					Photon(Ray(Vertex(), Vertex(1, -1, 0).normalized()),
						Color(0.125f, 0.125f, 0.5f, 1.0f)));
				scene.lights.cpuHandle()->operator[](1).use<SimpleDirectionalLight>(
					Photon(Ray(Vertex(), Vertex(-0.5f, -1, 0.580611f).normalized()),
						Color(0.125f, 0.5f, 0.125f, 1.0f)));
				scene.lights.cpuHandle()->operator[](2).use<SimpleDirectionalLight>(
					Photon(Ray(Vertex(), Vertex(-0.5f, -1, -0.580611f).normalized()),
						Color(0.5f, 0.125f, 0.125f, 1.0f)));
				camera.cpuHandle()->lense.use<DefaultPerspectiveLense>();
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

		BufferedRenderer *makeRenderer(
			const Renderer::ThreadConfiguration &configuration, void *contextAddr) {
			Context *context = ((Context*)contextAddr);
			DumbRenderer *renderer = new DumbRenderer(configuration);
			renderer->setScene(&context->scene);
			renderer->setCamera(&context->camera);
			return renderer;
		}
	}

	struct JNDB : public BakedTriFace {
		int index;
	};

	void testPerformance() {
		Context context;
		FrameBufferManager bufferA, bufferB;
		bufferA.cpuHandle()->use<BlockBuffer>();
		bufferB.cpuHandle()->use<BlockBuffer>();
		BufferedRenderProcessTest::runTestGauntlet(makeRenderer, ((void*)&context), 
			&bufferA, &bufferB, BufferedRenderProcessTest::TEST_RUN_FULL_GAUNTLET);
	}
}
