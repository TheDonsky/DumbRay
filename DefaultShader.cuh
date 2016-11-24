#pragma once
#include"Material.cuh"



class DefaultShader {
public:
	__dumb__ DefaultShader(ColorRGB color = ColorRGB(1, 1, 1), float diffuse = 0.2f, float smoothness = 0.8f, float shine = 16.0f);
	__dumb__ ShaderReport cast(const ShaderHitInfo<BakedTriFace> &input);

private:
	ColorRGB albedo;
	float diff;
	float gloss;
	float shininess;
};




#include"DefaultShader.impl.cuh"
