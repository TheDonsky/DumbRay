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

	__dumb__ void getPhoton(const void *lightSource, const Vertex &targetPoint, bool *noShadows, PhotonPack &result)const;
	__dumb__ ColorRGB ambient(const void *lightSource, const Vertex &targetPoint)const;

private:
	void(*getPhotonFunction)(const void *lightSource, const Vertex &targetPoint, bool *noShadows, PhotonPack &result);
	ColorRGB(*ambientFunction)(const void *lightSource, const Vertex &targetPoint);

	template<typename LightType>
	__dumb__ static void getPhotonAbstract(const void *lightSource, const Vertex &targetPoint, bool *noShadows, PhotonPack &result);
	template<typename LightType>
	__dumb__ static ColorRGB ambientAbstract(const void *lightSource, const Vertex &targetPoint);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class Light : public Generic<LightInterface> {
public:
	__dumb__ void getPhotons(const Vertex &targetPoint, bool *noShadows, PhotonPack &result)const;
	__dumb__ ColorRGB ambient(const Vertex &targetPoint)const;

	inline Light *upload()const;
	inline static Light* upload(const Light *source, int count = 1);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
SPECIALISE_TYPE_TOOLS_FOR(Light);




#include"Light.impl.cuh"

