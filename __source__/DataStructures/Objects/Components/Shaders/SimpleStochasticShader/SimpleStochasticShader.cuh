#pragma once
#include"../Material.cuh"
#include"../../../Meshes/BakedTriMesh/BakedTriMesh.h"
#include "../../../../DumbRenderContext/DumbRenderContext.cuh"



class SimpleStochasticShader {
public:
	__dumb__ SimpleStochasticShader(const ColorRGB &color = ColorRGB(1, 1, 1), float diffuse = 0.05f, float smoothness = 0.95f, float shininess = 2.0f);

	__dumb__ void requestIndirectSamples(const ShaderIndirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const;


	inline bool fromDson(const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context);


private:
	ColorRGB albedo;
	float diff, gloss, shine;
};


#include "SimpleStochasticShader.impl.cuh"
