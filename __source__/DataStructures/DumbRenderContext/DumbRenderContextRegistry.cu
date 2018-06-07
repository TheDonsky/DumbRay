#include "DumbRenderContext.cuh"


#include "../Objects/Components/Shaders/DumbBasedShader/DumbBasedShader.cuh"

void DumbRenderContext::registerMaterials() {
	registerMaterialType<DumbBasedShader>("dumb_based");
}


#include "../Objects/Scene/Lights/SimpleSoftDirectionalLight/SimpleSoftDirectionalLight.cuh"

void DumbRenderContext::registerLights() {
	registerLightType<SimpleSoftDirectionalLight>("simple_soft_directional");
}


#include "../Objects/Components/Lenses/SimpleStochasticLense/SimpleStochasticLense.cuh"

void DumbRenderContext::registerLenses() {
	registerLenseType<SimpleStochasticLense>("simple_stochastic");
}


