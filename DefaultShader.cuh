#pragma once
#include"Material.cuh"



class DefaultShader {
public:
	__device__ __host__ inline DefaultShader(ColorRGB color = ColorRGB(1, 1, 1), float diffuse = 0.2f, float smoothness = 0.8f, float shine = 16.0f);

	inline bool uploadAt(DefaultShader *dst);
	inline static bool disposeOnDevice(DefaultShader *ptr);

	__dumb__ Material<BakedTriFace>::ShaderReport cast(const Material<BakedTriFace>::HitInfo &input);

private:
	ColorRGB albedo;
	float diff;
	float gloss;
	float shininess;
};




#include"DefaultShader.impl.cuh"
