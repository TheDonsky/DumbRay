#pragma once

#include"../Material.cuh"
#include"../../../Meshes/BakedTriMesh/BakedTriMesh.h"
#include "../../../../DumbRenderContext/DumbRenderContext.cuh"


namespace Shaders {

	class FresnelShader {
	public:
		__dumb__ FresnelShader(const ColoredTexture &fresnel = ColoredTexture(Color(1.0f, 1.0f, 1.0f, 1.0f)));

		__dumb__ void requestIndirectSamples(const ShaderIndirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const;
		__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const;

		inline bool fromDson(const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context);

	private:
		ColoredTexture color;
	};

}

#include "FresnelShader.impl.cuh"
