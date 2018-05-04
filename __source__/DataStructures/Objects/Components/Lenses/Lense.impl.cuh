#include"Lense.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ LenseFunctionPack::LenseFunctionPack() {
	clean();
}
__dumb__ void LenseFunctionPack::clean() {
	getPixelSamplesFn = NULL;
	getPixelColorFn = NULL;
}
template<typename LenseType>
__dumb__ void LenseFunctionPack::use() {
	getPixelSamplesFn = getPixelSamplesGeneric<LenseType>;
	getPixelColorFn = getPixelColorGeneric<LenseType>;
}

__dumb__ void LenseFunctionPack::getPixelSamples(const void *lense, const Vector2 &screenSpacePosition, float pixelSize, RaySamples *samples)const {
	getPixelSamplesFn(lense, screenSpacePosition, pixelSize, samples);
}
__dumb__ Color LenseFunctionPack::getPixelColor(const void *lense, const Vector2 &screenSpacePosition, const Photon &photon)const {
	return getPixelColorFn(lense, screenSpacePosition, photon);
}

template<typename LenseType>
__dumb__ void LenseFunctionPack::getPixelSamplesGeneric(const void *lense, const Vector2 &screenSpacePosition, float pixelSize, RaySamples *samples) {
	((const LenseType*)lense)->getPixelSamples(screenSpacePosition, pixelSize, samples);
}

template<typename LenseType>
__dumb__ Color LenseFunctionPack::getPixelColorGeneric(const void *lense, const Vector2 &screenSpacePosition, const Photon &photon) {
	return ((const LenseType*)lense)->getPixelColor(screenSpacePosition, photon);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ void Lense::getPixelSamples(const Vector2 &screenSpacePosition, float pixelSize, RaySamples *samples)const {
	return functions().getPixelSamples(object(), screenSpacePosition, pixelSize, samples);
}
__dumb__ Color Lense::getPixelColor(const Vector2 &screenSpacePosition, const Photon &photon)const {
	return functions().getPixelColor(object(), screenSpacePosition, photon);
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





