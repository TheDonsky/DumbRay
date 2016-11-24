#pragma once

#include"Photon.cuh"
#include"Generic.cuh"



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
struct ShaderHitInfo {
	HitType object;
	Photon photon;
	Vector3 hitPoint;
	float distance;
	Vector3 observer;
};

struct ShaderReport {
	Photon observed;
	Photon reflection;
	Photon refraction;
	__dumb__ ShaderReport(const Photon &obs = Photon::zero(), const Photon &refl = Photon::zero(), const Photon &refr = Photon::zero());
};




/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
class Shader {
public:
	__dumb__ void clean();
	template<typename ShaderType>
	__dumb__ void use();

	__dumb__ ShaderReport cast(const void *shader, const ShaderHitInfo<HitType>& info)const;


private:
	ShaderReport(*castFunction)(const void *shader, const ShaderHitInfo<HitType>&info);
	template<typename ShaderType>
	__dumb__ static ShaderReport castGeneric(const void *shader, const ShaderHitInfo<HitType>& info);
};




/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType> class Material;
template<typename HitType>
class TypeTools<Material<HitType> > {
public:
	typedef Material<HitType> MaterialType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(MaterialType);
};



template<typename HitType>
class Material : public Generic<Shader<HitType> > {
public:
	__dumb__ ShaderReport cast(const ShaderHitInfo<HitType>& info)const;


	inline Material *upload()const;
	inline static Material* upload(const Material *source, int count = 1);
};





#include"Material.impl.cuh"
