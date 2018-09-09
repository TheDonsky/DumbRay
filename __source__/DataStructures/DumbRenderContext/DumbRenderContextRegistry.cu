#include "DumbRenderContextRegistry.cuh"


#include "../Objects/Components/Shaders/DumbBasedShader/DumbBasedShader.cuh"
#include "../Objects/Components/Shaders/SimpleStochasticShader/SimpleStochasticShader.cuh"
#include "../Objects/Components/Shaders/FresnelShader/FresnelShader.cuh"

void DumbRenderContextRegistry::registerMaterials() {
	registerMaterialType<DumbBasedShader>("dumb_based");
	registerMaterialType<SimpleStochasticShader>("simple_stochastic");
	registerMaterialType<Shaders::FresnelShader>("fresnel_color");
}


#include "../Objects/Scene/Lights/SimpleSoftDirectionalLight/SimpleSoftDirectionalLight.cuh"
#include "../Objects/Scene/Lights/SphericalLightEmitter/SphericalLightEmitter.cuh"
#include "../Objects/Scene/Lights/Spotlight/Spotlight.cuh"

void DumbRenderContextRegistry::registerLights() {
	registerLightType<SimpleSoftDirectionalLight>("simple_soft_directional");
	registerLightType<SphericalLightEmitter>("simple_spherical");
	registerLightType<Lights::Spotlight>("spotlight");
}


#include "../Objects/Components/Lenses/SimpleStochasticLense/SimpleStochasticLense.cuh"
#include "../Objects/Components/Lenses/SphericalSegmentLense/SphericalSegmentLense.cuh"

void DumbRenderContextRegistry::registerLenses() {
	registerLenseType<SimpleStochasticLense>("simple_stochastic");
	registerLenseType<Lenses::SphericalSegmentLense>("spherical_segment");
}


