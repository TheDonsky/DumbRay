#pragma once
#include"../Material.cuh"
#include"../../../Meshes/BakedTriMesh/BakedTriMesh.h"



class DumbBasedShader {
public:
	__dumb__ DumbBasedShader(const ColorRGB &fresnelFactor, float cpecular, const ColorRGB &diffuse, float metal);
	
	__dumb__ static DumbBasedShader roughGold() { return DumbBasedShader(ColorRGB(1.022f, 0.782f, 0.344f), 16.0f, ColorRGB(0.0f, 0.0f, 0.0f), 1.0f); }
	__dumb__ static DumbBasedShader glossyFinish() { return DumbBasedShader(ColorRGB(0.175f, 0.15f, 0.125f), 4096.0f, ColorRGB(0.32f, 0.32f, 0.32f), 0.25f); }
	__dumb__ static DumbBasedShader matteFinish() { return DumbBasedShader(ColorRGB(0.175f, 0.15f, 0.125f), 8.0f, ColorRGB(0.24f, 0.24f, 0.24f), 0.05f); }

	__dumb__ void requestIndirectSamples(const ShaderIndirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const;



private:
	ColorRGB fres;
	ColorRGB diff;
	float spec;
	float specMass;

	__dumb__ static float fresnel(float r, const Vector3 &wh, const Vector3 &wi);
};

#include "DumbBasedShader.impl.cuh"
