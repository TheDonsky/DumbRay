#pragma once
#include"Photon.cuh"
#include"Vector2.h"
#include"Generic.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class LenseFunctionPack {
public:
	__dumb__ LenseFunctionPack();
	__dumb__ void clean();
	template<typename LenseType>
	__dumb__ void use();


	__dumb__ Photon getScreenPhoton(const void *lense, const Vector2 &screenSpacePosition)const;
	__dumb__ Photon toScreenSpace(const void *lense, const Photon &photon)const;


private:
	Photon(*getScreenPhotonFunction)(const void* lense, const Vector2 &screenSpacePosition);
	Photon(*toScreenSpaceFunction)(const void* lense, const Photon &photon);

	template<typename LenseType>
	__dumb__ static Photon getScreenPhotonGeneric(const void* lense, const Vector2 &screenSpacePosition);
	template<typename LenseType>
	__dumb__ static Photon toScreenSpaceGeneric(const void* lense, const Photon &photon);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class Lense : public Generic<LenseFunctionPack>{
public:
	__dumb__ Photon getScreenPhoton(const Vector2 &screenSpacePosition)const;
	__dumb__ Photon toScreenSpace(const Photon &photon)const;

	inline Lense *upload()const;
	inline static Lense* upload(const Lense *source, int count = 1);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
SPECIALISE_TYPE_TOOLS__FOR(Lense);




#include"Lense.impl.cuh"
