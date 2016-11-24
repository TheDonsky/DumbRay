#pragma once
#include"Material.cuh"
#include"BakedTriMesh.h"



class DummyShader {
public:
	__dumb__ ShaderReport cast(const ShaderHitInfo<BakedTriFace> &input);

private:
	int garbage;
};



#include"DummyShader.impl.cuh"
