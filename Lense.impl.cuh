#include"Lense.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ LenseFunctionPack::LenseFunctionPack() {
	clean();
}
__dumb__ void LenseFunctionPack::clean() {
	getScreenPhotonFunction = NULL;
	toScreenSpaceFunction = NULL;
}
template<typename LenseType>
__dumb__ void LenseFunctionPack::use() {
	getScreenPhotonFunction = getScreenPhotonGeneric<LenseType>;
	toScreenSpaceFunction = toScreenSpaceGeneric<LenseType>;
}


__dumb__ Photon LenseFunctionPack::getScreenPhoton(const void *lense, const Vector2 &screenSpacePosition)const {
	return getScreenPhotonFunction(lense, screenSpacePosition);
}
__dumb__ Photon LenseFunctionPack::toScreenSpace(const void *lense, const Photon &photon)const {
	return toScreenSpaceFunction(lense, photon);
}


template<typename LenseType>
__dumb__ Photon LenseFunctionPack::getScreenPhotonGeneric(const void* lense, const Vector2 &screenSpacePosition) {
	return ((LenseType*)lense)->getScreenPhoton(screenSpacePosition);
}
template<typename LenseType>
__dumb__ Photon LenseFunctionPack::toScreenSpaceGeneric(const void* lense, const Photon &photon) {
	return ((LenseType*)lense)->toScreenSpace(photon);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ Photon Lense::getScreenPhoton(const Vector2 &screenSpacePosition)const {
	return functions().getScreenPhoton(object(), screenSpacePosition);
}
__dumb__ Photon Lense::toScreenSpace(const Photon &photon)const {
	return functions().toScreenSpace(object(), photon);
}


inline Lense* Lense::upload()const {
	return ((Lense*)Generic<LenseFunctionPack>::upload());
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
COPY_TYPE_TOOLS_IMPLEMENTATION(Lense, Generic<LenseFunctionPack>);





