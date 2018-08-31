#pragma once

#include"../Material.cuh"
#include"../../../Meshes/BakedTriMesh/BakedTriMesh.h"
#include "../../../../DumbRenderContext/DumbRenderContext.cuh"


namespace Shaders {

	class GaussianShader {
	public:
		__dumb__ GaussianShader();


		__dumb__ void requestIndirectSamples(const ShaderIndirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const;
		__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const;

		inline bool fromDson(const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context);

	private:
	};

}

#include "GaussianShader.impl.cuh"
