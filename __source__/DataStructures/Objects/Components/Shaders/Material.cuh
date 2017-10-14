#pragma once
#include"../../../Primitives/Compound/Photon/Photon.cuh"
#include"../../../GeneralPurpose/Generic/Generic.cuh"
#include"../../Components/DumbStructs.cuh"



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
struct ShaderBounceInfo {
	const HitType *object;
	Photon photon;
	Vector3 hitPoint;
};

template<typename HitType>
struct ShaderHitInfo {
	const HitType *object;
	Photon photon;
	Vector3 hitPoint;
	Vector3 observer;
};

struct ShaderReport {
	Photon observed;
	Photon bounce;
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

	//__dumb__ ShaderReport cast(const void *shader, const ShaderHitInfo<HitType>& info)const;
	__dumb__ void bounce(const void *shader, const ShaderBounceInfo<HitType> &info, PhotonPack &result)const;
	__dumb__ Photon illuminate(const void *shader, const ShaderHitInfo<HitType>& info)const;


private:
	//ShaderReport(*castFunction)(const void *shader, const ShaderHitInfo<HitType>&info);
	void(*bounceFunction)(const void *shader, const ShaderBounceInfo<HitType> &info, PhotonPack &result);
	Photon(*illuminateFunction)(const void *shader, const ShaderHitInfo<HitType>&info);
	/*
	template<typename ShaderType>
	__dumb__ static ShaderReport castGeneric(const void *shader, const ShaderHitInfo<HitType>& info);
	//*/
	template<typename ShaderType>
	__dumb__ static void bounceGeneric(const void *shader, const ShaderBounceInfo<HitType> &info, PhotonPack &result);
	template<typename ShaderType>
	__dumb__ static Photon illuminateGeneric(const void *shader, const ShaderHitInfo<HitType>& info);
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
	//__dumb__ ShaderReport cast(const ShaderHitInfo<HitType>& info)const;
	__dumb__ void bounce(const ShaderBounceInfo<HitType> &info, PhotonPack &result)const;
	__dumb__ Photon illuminate(const ShaderHitInfo<HitType>& info)const;


	inline Material *upload()const;
	inline static Material* upload(const Material *source, int count = 1);
};





#include"Material.impl.cuh"
