#pragma once
#include"../Material.cuh"
#include"../../../Meshes/BakedTriMesh/BakedTriMesh.h"



class DumbBasedShader {
public:
	/*
	__dumb__ DumbBasedShader(
		const ColorRGB &fresnelFactor = ColorRGB(1.022f, 0.782f, 0.344f), 
		float cpecular = 16.0f, 
		const ColorRGB &diffuse = ColorRGB(0.0f, 0.0f, 0.0f), 
		float metal = 1.0f);
	/*/
	__dumb__ DumbBasedShader(
		const ColorRGB &fresnelFactor = ColorRGB(0.175f, 0.15f, 0.125f), 
		float cpecular = 4096.0f, 
		const ColorRGB &diffuse = ColorRGB(0.32f, 0.32f, 0.32f), 
		float metal = 0.25f);
	//*/

	__dumb__ void requestIndirectSamples(const ShaderInirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const;



private:
	ColorRGB fres;
	ColorRGB diff;
	float spec;
	float specMass;

	__dumb__ static float fresnel(float r, const Vector3 &wh, const Vector3 &wi);
};


#include "DumbBasedShader.impl.cuh"
