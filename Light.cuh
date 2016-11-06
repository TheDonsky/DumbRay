#pragma once
#include"Generic.cuh"
#include"Vector3.h"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename IlluminatedType>
class LightInterface {
public:
	__dumb__ void clean();
	template<typename LenseType>
	__dumb__ void use();

private:
	Photon(*photonToFunction)(const Vertex &v);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename IlluminatedType> class Light;
template<typename IlluminatedType>
class TypeTools<Light<IlluminatedType> > {
public:
	typedef Light<IlluminatedType> LightType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(LightType);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename IlluminatedType>
class Light : public Generic<LightInterface<IlluminatedType> > {
public:

	inline Light *upload()const;
	inline static Light* upload(const Light *source, int count = 1);
};






#include"Light.impl.cuh"

