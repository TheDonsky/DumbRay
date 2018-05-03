#pragma once
#include"../../../GeneralPurpose/Generic/Generic.cuh"
#include"../../../Primitives/Compound/Photon/Photon.cuh"
#include"../../../Primitives/Pure/Vector3/Vector3.h"
#include"../../Components/DumbStructs.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class LightInterface {
public:
	__dumb__ void clean();
	template<typename LightType>
	__dumb__ void use();

	__dumb__ void getVertexPhotons(const void *lightSource, const Vector3 &point, PhotonSamples *result, bool *castShadows)const;



private:
	void(*getVertexPhotonsFn)(const void *lightSource, const Vector3 &point, PhotonSamples *result, bool *castShadows);

	template<typename LightType>
	__dumb__ static void getVertexPhotonsAbstract(const void *lightSource, const Vector3 &point, PhotonSamples *result, bool *castShadows);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class Light : public Generic<LightInterface> {
public:
	__dumb__ void getVertexPhotons(const Vector3 &point, PhotonSamples *result, bool *castShadows)const;

	inline Light *upload()const;
	inline static Light* upload(const Light *source, int count = 1);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
SPECIALISE_TYPE_TOOLS_FOR(Light);




#include"Light.impl.cuh"

