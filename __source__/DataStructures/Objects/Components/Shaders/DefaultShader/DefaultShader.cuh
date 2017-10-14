#pragma once
#include"../Material.cuh"



template<typename HitType = BakedTriFace>
class DefaultShaderGeneric {
public:
	__dumb__ DefaultShaderGeneric(ColorRGB color = ColorRGB(1, 1, 1), float diffuse = 0.7f, float smoothness = 0.2f, float shine = 8.0f);
	__dumb__ ShaderReport cast(const ShaderHitInfo<HitType> &input)const;
	__dumb__ void bounce(const ShaderBounceInfo<HitType> &info, PhotonPack &result)const;
	__dumb__ Photon illuminate(const ShaderHitInfo<HitType>& info)const;

private:
	ColorRGB albedo;
	float diff;
	float gloss;
	float shininess;
};

typedef DefaultShaderGeneric<> DefaultShader;



#include"DefaultShader.impl.cuh"
