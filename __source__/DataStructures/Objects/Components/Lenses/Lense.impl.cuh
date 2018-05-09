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

__dumb__ void LenseFunctionPack::getPixelSamples(const void *lense, const LenseGetPixelSamplesRequest &request, RaySamples *samples)const {
	getPixelSamplesFn(lense, request, samples);
}
__dumb__ Color LenseFunctionPack::getPixelColor(const void *lense, const LenseGetPixelColorRequest &request)const {
	return getPixelColorFn(lense, request);
}

template<typename LenseType>
__dumb__ void LenseFunctionPack::getPixelSamplesGeneric(const void *lense, const LenseGetPixelSamplesRequest &request, RaySamples *samples) {
	((const LenseType*)lense)->getPixelSamples(request, samples);
}

template<typename LenseType>
__dumb__ Color LenseFunctionPack::getPixelColorGeneric(const void *lense, const LenseGetPixelColorRequest &request) {
	return ((const LenseType*)lense)->getPixelColor(request);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ void Lense::getPixelSamples(const LenseGetPixelSamplesRequest &request, RaySamples *samples)const {
	return functions().getPixelSamples(object(), request, samples);
}
__dumb__ Color Lense::getPixelColor(const LenseGetPixelColorRequest &request)const {
	return functions().getPixelColor(object(), request);
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





