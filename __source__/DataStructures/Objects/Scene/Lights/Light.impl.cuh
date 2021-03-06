#include"Light.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ void LightInterface::clean() {
	getVertexPhotonsFn = NULL;
}
template<typename LightType>
__dumb__ void LightInterface::use() {
	getVertexPhotonsFn = getVertexPhotonsAbstract<LightType>;
}

__dumb__ void LightInterface::getVertexPhotons(const void *lightSource, const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const {
	getVertexPhotonsFn(lightSource, request, result, castShadows);
}

template<typename LightType>
__dumb__ void LightInterface::getVertexPhotonsAbstract(const void *lightSource, const LightVertexSampleRequest request, PhotonSamples *result, bool *castShadows) {
	((LightType*)lightSource)->getVertexPhotons(request, result, castShadows);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ void Light::getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const {
	return functions().getVertexPhotons(object(), request, result, castShadows);
}

inline Light* Light::upload()const {
	return (Light*)(Generic<LightInterface>::upload());
}
inline Light* Light::upload(const Light *source, int count) {
	return (Light*)(Generic<LightInterface>::upload(source, count));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
COPY_TYPE_TOOLS_IMPLEMENTATION(Light, Generic<LightInterface>);





