#include "DumbRenderContext.cuh"


#include "../Objects/Components/Shaders/DumbBasedShader/DumbBasedShader.cuh"
#include "../Objects/Components/Shaders/SimpleStochasticShader/SimpleStochasticShader.cuh"

void DumbRenderContext::registerMaterials() {
	registerMaterialType<DumbBasedShader>("dumb_based");
	registerMaterialType<SimpleStochasticShader>("simple_stochastic");
}


#include "../Objects/Scene/Lights/SimpleSoftDirectionalLight/SimpleSoftDirectionalLight.cuh"
#include "../Objects/Scene/Lights/SphericalLightEmitter/SphericalLightEmitter.cuh"

void DumbRenderContext::registerLights() {
	registerLightType<SimpleSoftDirectionalLight>("simple_soft_directional");
	registerLightType<SphericalLightEmitter>("simple_spherical");
}


#include "../Objects/Components/Lenses/SimpleStochasticLense/SimpleStochasticLense.cuh"

void DumbRenderContext::registerLenses() {
	registerLenseType<SimpleStochasticLense>("simple_stochastic");
}


