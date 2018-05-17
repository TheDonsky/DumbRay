#pragma once
#include"../Material.cuh"
#include"../../../Meshes/BakedTriMesh/BakedTriMesh.h"



class DummyShader {
public:
	__dumb__ void requestIndirectSamples(const ShaderIndirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const;


private:
	int garbage;
};



#include"DummyShader.impl.cuh"
