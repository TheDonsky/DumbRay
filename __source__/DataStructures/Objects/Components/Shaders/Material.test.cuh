#pragma once
#include "Material.cuh"
#include "DummyShader/DummyShader.cuh"
#include "../../../../Namespaces/Tests/Tests.h"


namespace MaterialTest {
	namespace Private {
		__dumb__ void makeMaterialShout(const Material<BakedTriFace> &material) {
			RaySamples samples;
			material.requestIndirectSamples(ShaderInirectSamplesRequest<BakedTriFace>(), &samples);
		}
		__global__ void materialShoutFromKernel(const Material<BakedTriFace> *material) {
			makeMaterialShout(*material);
		}
		inline static void test() {
			Material<BakedTriFace> material;
			material.use<DummyShader>();
			Material<BakedTriFace> *materialCopy = material.upload();
			materialShoutFromKernel<<<1, 1>>>(materialCopy);
			cudaDeviceSynchronize();
			Material<BakedTriFace>::dispose(materialCopy);
			cudaFree(materialCopy);
			makeMaterialShout(material);
			material.clean();
		}
	}

	inline static void test() {
		Tests::runTest(Private::test, "Testing Material...");
	}
}
