#pragma once
#include"../Material.cuh"
#include"../../../Meshes/BakedTriMesh/BakedTriMesh.h"



class DummyShader {
public:
	__dumb__ ShaderReport cast(const ShaderHitInfo<BakedTriFace> &input)const;
	__dumb__ void bounce(const ShaderBounceInfo<BakedTriFace> &info, PhotonPack &result)const;
	__dumb__ Photon illuminate(const ShaderHitInfo<BakedTriFace>& info)const;

	__dumb__ void requestIndirectSamples(const ShaderInirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const;


private:
	int garbage;
};



#include"DummyShader.impl.cuh"
