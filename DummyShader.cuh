#pragma once
#include"Material.cuh"
#include"BakedTriMesh.h"



class DummyShader {
public:

	inline bool uploadAt(DummyShader *dst);
	inline static bool disposeOnDevice(DummyShader *ptr);

	__dumb__ Material<BakedTriFace>::ShaderReport cast(const Material<BakedTriFace>::HitInfo &input);

private:
	int garbage;
};



#include"DummyShader.impl.cuh"
