#pragma once
#include"../../../Primitives/Compound/Photon/Photon.cuh"
#include"../../../Primitives/Pure/Vector2/Vector2.h"
#include"../../../GeneralPurpose/Generic/Generic.cuh"
#include"../../../Objects/Components/DumbStructs.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class LenseFunctionPack {
public:
	__dumb__ LenseFunctionPack();
	__dumb__ void clean();
	template<typename LenseType>
	__dumb__ void use();

	__dumb__ void getPixelSamples(const void *lense, const Vector2 &screenSpacePosition, float pixelSize, RaySamples *samples)const;
	__dumb__ Color getPixelColor(const void *lense, const Vector2 &screenSpacePosition, const Photon &photon)const;



private:
	void(*getPixelSamplesFn)(const void *lense, const Vector2 &screenSpacePosition, float pixelSize, RaySamples *samples);
	Color(*getPixelColorFn)(const void *lense, const Vector2 &screenSpacePosition, const Photon &photon);

	template<typename LenseType>
	__dumb__ static void getPixelSamplesGeneric(const void *lense, const Vector2 &screenSpacePosition, float pixelSize, RaySamples *samples);
	template<typename LenseType>
	__dumb__ static Color getPixelColorGeneric(const void *lense, const Vector2 &screenSpacePosition, const Photon &photon);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class Lense : public Generic<LenseFunctionPack>{
public:
	__dumb__ void getPixelSamples(const Vector2 &screenSpacePosition, float pixelSize, RaySamples *samples)const;
	__dumb__ Color getPixelColor(const Vector2 &screenSpacePosition, const Photon &photon)const;

	inline Lense *upload()const;
	inline static Lense* upload(const Lense *source, int count = 1);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
SPECIALISE_TYPE_TOOLS_FOR(Lense);




#include"Lense.impl.cuh"
