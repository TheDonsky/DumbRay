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
	getColorFunction = NULL;
}
template<typename LenseType>
__dumb__ void LenseFunctionPack::use() {
	getScreenPhotonFunction = getScreenPhotonGeneric<LenseType>;
	toScreenSpaceFunction = toScreenSpaceGeneric<LenseType>;
	getColorFunction = getColorGeneric<LenseType>;
	/*
	printf("LENSE FUNCTION PACK(%p): ", this);
	for (size_t i = 0; i < sizeof(LenseFunctionPack); i++)
		printf(" %02X", ((char*)((void*)this))[i] & 0xFF);
	printf("\n");
	//*/
}


__dumb__ void LenseFunctionPack::getScreenPhoton(const void *lense, const Vector2 &screenSpacePosition, PhotonPack &result)const {
	getScreenPhotonFunction(lense, screenSpacePosition, result);
}
__dumb__ Photon LenseFunctionPack::toScreenSpace(const void *lense, const Photon &photon)const {
	return toScreenSpaceFunction(lense, photon);
}
__dumb__ void LenseFunctionPack::getColor(const void *lense, const Vector2 &screenSpacePosition, Photon photon, Color &result)const {
	return getColorFunction(lense, screenSpacePosition, photon, result);
}


template<typename LenseType>
__dumb__ void LenseFunctionPack::getScreenPhotonGeneric(const void* lense, const Vector2 &screenSpacePosition, PhotonPack &result) {
	((LenseType*)lense)->getScreenPhoton(screenSpacePosition, result);
}
template<typename LenseType>
__dumb__ Photon LenseFunctionPack::toScreenSpaceGeneric(const void* lense, const Photon &photon) {
	return ((LenseType*)lense)->toScreenSpace(photon);
}
template<typename LenseType>
__dumb__ void LenseFunctionPack::getColorGeneric(const void *lense, const Vector2 &screenSpacePosition, Photon photon, Color &result) {
	return ((LenseType*)lense)->getColor(screenSpacePosition, photon, result);
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
__dumb__ void Lense::getColor(const Vector2 &screenSpacePosition, Photon photon, Color &result)const {
	return functions().getColor(object(), screenSpacePosition, photon, result);
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





