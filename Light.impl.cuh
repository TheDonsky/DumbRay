#include"Light.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ void LightInterface::clean() {
	getPhotonFunction = NULL;
	ambientFunction = NULL;
}
template<typename LightType>
__dumb__ void LightInterface::use() {
	getPhotonFunction = getPhotonAbstract<LightType>;
	ambientFunction = ambientAbstract<LightType>;
}

__dumb__ Photon LightInterface::getPhoton(const void *lightSource, const Vertex &targetPoint, bool *noShadows) const {
	return getPhotonFunction(lightSource, targetPoint, noShadows);
}
__dumb__ ColorRGB LightInterface::ambient(const void *lightSource, const Vertex &targetPoint) const {
	return ambientFunction(lightSource, targetPoint);
}

template<typename LightType>
__dumb__ Photon LightInterface::getPhotonAbstract(const void *lightSource, const Vertex &targetPoint, bool *noShadows) {
	return ((LightType*)lightSource)->getPhoton(targetPoint, noShadows);
}
template<typename LightType>
__dumb__ ColorRGB LightInterface::ambientAbstract(const void *lightSource, const Vertex &targetPoint) {
	return ((LightType*)lightSource)->ambient(targetPoint);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ Photon Light::getPhoton(const Vertex &targetPoint, bool *noShadows) const {
	return functions().getPhoton(object(), targetPoint, noShadows);
}
__dumb__ ColorRGB Light::ambient(const Vertex &targetPoint) const {
	return functions().ambient(object(), targetPoint);
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





