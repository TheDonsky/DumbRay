#pragma once
#include"Generic.cuh"
#include"Photon.cuh"
#include"Vector3.h"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class LightInterface {
public:
	__dumb__ void clean();
	template<typename LightType>
	__dumb__ void use();

	__dumb__ Photon getPhoton(const void *lightSource, const Vertex &targetPoint, bool *noShadows)const;
	__dumb__ ColorRGB ambient(const void *lightSource, const Vertex &targetPoint)const;

private:
	Photon(*getPhotonFunction)(const void *lightSource, const Vertex &targetPoint, bool *noShadows);
	ColorRGB(*ambientFunction)(const void *lightSource, const Vertex &targetPoint);

	template<typename LightType>
	__dumb__ static Photon getPhotonAbstract(const void *lightSource, const Vertex &targetPoint, bool *noShadows);
	template<typename LightType>
	__dumb__ static ColorRGB ambientAbstract(const void *lightSource, const Vertex &targetPoint);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class Light : public Generic<LightInterface> {
public:
	__dumb__ Photon getPhoton(const Vertex &targetPoint, bool *noShadows)const;
	__dumb__ ColorRGB ambient(const Vertex &targetPoint)const;

	inline Light *upload()const;
	inline static Light* upload(const Light *source, int count = 1);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
SPECIALISE_TYPE_TOOLS_FOR(Light);




#include"Light.impl.cuh"

