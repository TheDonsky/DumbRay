#pragma once
#include"Material.cuh"
#include"BakedTriMesh.h"



class DummyShader {
public:
	__dumb__ ShaderReport cast(const ShaderHitInfo<BakedTriFace> &input)const;
	__dumb__ void bounce(const ShaderBounceInfo<BakedTriFace> &info, ShaderBounce *bounce)const;
	__dumb__ Photon illuminate(const ShaderHitInfo<BakedTriFace>& info)const;

private:
	int garbage;
};



#include"DummyShader.impl.cuh"
