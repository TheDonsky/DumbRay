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


__dumb__ void LenseFunctionPack::getScreenPhoton(const void *lense, const Vector2 &screenSpacePosition, PhotonPack &result)const {
	getScreenPhotonFunction(lense, screenSpacePosition, result);
}
__dumb__ Photon LenseFunctionPack::toScreenSpace(const void *lense, const Photon &photon)const {
	return toScreenSpaceFunction(lense, photon);
}


template<typename LenseType>
__dumb__ void LenseFunctionPack::getScreenPhotonGeneric(const void* lense, const Vector2 &screenSpacePosition, PhotonPack &result) {
	((LenseType*)lense)->getScreenPhoton(screenSpacePosition, result);
}
template<typename LenseType>
__dumb__ Photon LenseFunctionPack::toScreenSpaceGeneric(const void* lense, const Photon &photon) {
	return ((LenseType*)lense)->toScreenSpace(photon);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ void Lense::getScreenPhoton(const Vector2 &screenSpacePosition, PhotonPack &result)const {
	functions().getScreenPhoton(object(), screenSpacePosition, result);
}
__dumb__ Photon Lense::toScreenSpace(const Photon &photon)const {
	return functions().toScreenSpace(object(), photon);
}


inline Lense* Lense::upload()const {
	return ((Lense*)Generic<LenseFunctionPack>::upload());
}
inline Lense* Lense::upload(const Lense *source, int count) {
	return ((Lense*)Generic<LenseFunctionPack>::upload(source, count));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
COPY_TYPE_TOOLS_IMPLEMENTATION(Lense, Generic<LenseFunctionPack>);





