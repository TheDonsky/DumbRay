#include "FresnelShader.cuh"


namespace Shaders {

	__dumb__ FresnelShader::FresnelShader(const ColoredTexture &fresnel) {
		color = fresnel;
	}


	__dumb__ void FresnelShader::requestIndirectSamples(const ShaderIndirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const {
		samples->sampleCount = 0;
	}
	__dumb__ Color FresnelShader::getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const {
		Vector3 masses = request.object->vert.getMasses(request.hitPoint);
		Vector3 light = -request.photon.ray.direction.normalized();
		Vector3 normal = request.object->norm.massCenter(masses);
		return DumbTools::fresnel(color(request.object->tex.massCenter(masses), request.context), normal, light) * request.photon.color;
	}

	inline bool FresnelShader::fromDson(const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context) {
		color = ColoredTexture(Color(1.0f, 1.0f, 1.0f, 1.0f));
		if (!DumbTools::Colors::getColor(object, errorStream, "engine_color", color.color, NULL));
		return color.fromDson(object, errorStream, context);
	}

}

