#pragma once
#include"../Material.cuh"
#include"../../../Meshes/BakedTriMesh/BakedTriMesh.h"
#include "../../../../DumbRenderContext/DumbRenderContext.cuh"



class DumbBasedShader {
public:
	__dumb__ DumbBasedShader(
		const ColoredTexture &fresnelFactor = (ColoredTexture)ColorRGB(0.2f, 0.2f, 0.2f), float cpecular = 8.0f, 
		const ColoredTexture &diffuse = (ColoredTexture)ColorRGB(0.2f, 0.2f, 0.2f), float metal = 0.5f, const ColoredTexture &normal = ColoredTexture(ColorRGB(1.0f, 1.0f, 1.0f)));
	
	__dumb__ static DumbBasedShader roughGold() { return DumbBasedShader((ColoredTexture)ColorRGB(1.022f, 0.782f, 0.344f), 16.0f, (ColoredTexture)ColorRGB(1.0f, 1.0f, 1.0f), 1.0f); }
	__dumb__ static DumbBasedShader glossyGold() { return DumbBasedShader((ColoredTexture)ColorRGB(1.022f, 0.782f, 0.344f), 128.0f, (ColoredTexture)ColorRGB(1.0f, 1.0f, 1.0f), 1.0f); }
	__dumb__ static DumbBasedShader glossyFinish() { return DumbBasedShader((ColoredTexture)ColorRGB(0.175f, 0.15f, 0.125f), 4096.0f, (ColoredTexture)ColorRGB(0.32f, 0.32f, 0.32f), 0.25f); }
	__dumb__ static DumbBasedShader matteFinish() { return DumbBasedShader((ColoredTexture)ColorRGB(0.175f, 0.15f, 0.125f), 8.0f, (ColoredTexture)ColorRGB(0.24f, 0.24f, 0.24f), 0.05f); }

	__dumb__ void requestIndirectSamples(const ShaderIndirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const;

	inline bool fromDson(const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context);

private:
	ColoredTexture fresnelColor;
	ColoredTexture diffuseColor;
	ColoredTexture normalColor;
	ColoredTexture alphaCutout;
	float alphaCutoutThreshold;
	float spec;
	float specMass;
};

#include "DumbBasedShader.impl.cuh"
