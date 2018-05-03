#pragma once
#include"../Material.cuh"



template<typename HitType = BakedTriFace>
class DefaultShaderGeneric {
public:
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

	__dumb__ DefaultShaderGeneric(ColorRGB color = ColorRGB(1, 1, 1), float diffuse = 0.7f, float smoothness = 0.2f, float shine = 8.0f);
	__dumb__ ShaderReport cast(const ShaderHitInfo &input)const;

	__dumb__ void requestIndirectSamples(const ShaderInirectSamplesRequest<HitType> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<HitType> &request)const;


private:
	ColorRGB albedo;
	float diff;
	float gloss;
	float shininess;
};

typedef DefaultShaderGeneric<> DefaultShader;



#include"DefaultShader.impl.cuh"
