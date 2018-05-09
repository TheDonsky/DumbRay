#pragma once
#include"../../../Primitives/Compound/Photon/Photon.cuh"
#include"../../../Primitives/Pure/Vector2/Vector2.h"
#include"../../../GeneralPurpose/Generic/Generic.cuh"
#include"../../../Objects/Components/DumbStructs.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
struct LenseGetPixelSamplesRequest {
	// Screen space position of the pixel:
	Vector2 screenSpacePosition;

	// Relative on screen size of the pixel:
	float pixelSize;

	// The render context in all of it's glory:
	RenderContext *context;
};
struct LenseGetPixelColorRequest {
	// Screen space position of the pixel:
	Vector2 screenSpacePosition;

	// Relative on screen size of the pixel:
	float pixelSize;

	// Photon, that was cast upon the pixel:
	Photon photon;

	// Type of the photon:
	PhotonType photonType;

	// The render context in all of it's glory:
	RenderContext *context;
};



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class LenseFunctionPack {
public:
	__dumb__ LenseFunctionPack();
	__dumb__ void clean();
	template<typename LenseType>
	__dumb__ void use();

	__dumb__ void getPixelSamples(const void *lense, const LenseGetPixelSamplesRequest &request, RaySamples *samples)const;
	__dumb__ Color getPixelColor(const void *lense, const LenseGetPixelColorRequest &request)const;



private:
	void(*getPixelSamplesFn)(const void *lense, const LenseGetPixelSamplesRequest &request, RaySamples *samples);
	Color(*getPixelColorFn)(const void *lense, const LenseGetPixelColorRequest &request);

	template<typename LenseType>
	__dumb__ static void getPixelSamplesGeneric(const void *lense, const LenseGetPixelSamplesRequest &request, RaySamples *samples);
	template<typename LenseType>
	__dumb__ static Color getPixelColorGeneric(const void *lense, const LenseGetPixelColorRequest &request);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class Lense : public Generic<LenseFunctionPack>{
public:
	__dumb__ void getPixelSamples(const LenseGetPixelSamplesRequest &request, RaySamples *samples)const;
	__dumb__ Color getPixelColor(const LenseGetPixelColorRequest &request)const;

	inline Lense *upload()const;
	inline static Lense* upload(const Lense *source, int count = 1);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
SPECIALISE_TYPE_TOOLS_FOR(Lense);




#include"Lense.impl.cuh"
