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


	__dumb__ void getScreenPhoton(const void *lense, const Vector2 &screenSpacePosition, PhotonPack &result)const;
	__dumb__ Photon toScreenSpace(const void *lense, const Photon &photon)const;


private:
	void(*getScreenPhotonFunction)(const void* lense, const Vector2 &screenSpacePosition, PhotonPack &result);
	Photon(*toScreenSpaceFunction)(const void* lense, const Photon &photon);

	template<typename LenseType>
	__dumb__ static void getScreenPhotonGeneric(const void* lense, const Vector2 &screenSpacePosition, PhotonPack &result);
	template<typename LenseType>
	__dumb__ static Photon toScreenSpaceGeneric(const void* lense, const Photon &photon);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class Lense : public Generic<LenseFunctionPack>{
public:
	__dumb__ void getScreenPhoton(const Vector2 &screenSpacePosition, PhotonPack &result)const;
	__dumb__ Photon toScreenSpace(const Photon &photon)const;

	inline Lense *upload()const;
	inline static Lense* upload(const Lense *source, int count = 1);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
SPECIALISE_TYPE_TOOLS_FOR(Lense);




#include"Lense.impl.cuh"
