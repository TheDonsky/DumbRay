#pragma once
#include"../Material.cuh"
#include"../../../Meshes/BakedTriMesh/BakedTriMesh.h"



class DumbBasedShader {
public:
	__dumb__ DumbBasedShader(const ColorRGB &fresnelFactor = ColorRGB(1.022f, 0.782f, 0.344f), float cpecular = 16.0f);

	__dumb__ void requestIndirectSamples(const ShaderInirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const;



private:
	ColorRGB fres;
	float spec;

	__dumb__ static float fresnel(float r, const Vector3 &wh, const Vector3 &wi);
};


#include "DumbBasedShader.impl.cuh"
