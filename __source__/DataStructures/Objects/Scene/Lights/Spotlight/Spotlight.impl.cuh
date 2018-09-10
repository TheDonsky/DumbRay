#include "Spotlight.cuh"



namespace Lights {

	__dumb__ Spotlight::Spotlight(
		Color shade, float lum, Vertex pos, Vector3 dir,
		float innerAngle, float outerAngle, float falloffPower, 
		float discSize, bool castShadows, int samples) {
		
		color = shade;
		color *= lum;
		color.a = 1.0f;

		position = pos;
		direction = dir.normalized();


		if (discSize < (32 * VECTOR_EPSILON)) discSize = (32 * VECTOR_EPSILON);
		emitterSize = discSize;

		if (innerAngle < 0.0f) innerAngle = (-innerAngle);
		if (innerAngle > 180.0f) innerAngle = 180.0f;

		if (outerAngle < 0.0f) outerAngle = (-innerAngle);
		if (outerAngle > 180.0f) outerAngle = 180.0f;
		if (outerAngle < innerAngle) outerAngle = innerAngle;

		innerCosine = cos(innerAngle * PI / 180.0f);
		outerCosine = cos(outerAngle * PI / 180.0f);

		outerFalloff = falloffPower;

		flags = ((samples > 0) ? ((samples > 16) ? 16 : samples) : 0);
		if (castShadows) flags |= FLAGS_CAST_SHADOWS;
	}

	__dumb__ void Spotlight::getVertexPhotons(
		const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const {
		(*castShadows) = ((flags & FLAGS_CAST_SHADOWS) != 0);
		
		int resultCount = (((int)flags) & FLAGS_SAMPLE_COUNT_MASK);
		
		Vector3 nonDir = ((direction.z > 0.9f) ? Vector3(1.0f, 0.0f, 0.0f) : Vector3(0.0f, 0.0f, 1.0f));
		Vector3 ver = (direction & nonDir).normalized();
		Vector3 hor = (direction & ver);
		
		result->sampleCount = 0;
		for (int i = 0; i < resultCount; i++) {
			Vector3 pos;
			{
				float angle = request.context->entropy->range(0.0f, 2.0f * PI);
				pos = (position + (((ver * cos(angle)) + (hor * sin(angle))) * emitterSize * sqrt(request.context->entropy->getFloat())));
			}

			Vector3 delta = (request.point - pos);
			float sqrDistance = delta.sqrMagnitude();
			
			if (sqrDistance < VECTOR_EPSILON) continue;
			register float distance = sqrt(sqrDistance);

			Color col;
			{
				register float baseSurface = (emitterSize * emitterSize);
				register float spreadFactor = (1.0f - (innerCosine * innerCosine));
				register float surface = (baseSurface + (distance * spreadFactor));
				col = (color * (baseSurface / (((float)resultCount) * surface)));
			}

			Vector3 dir = (delta / distance);
			
			float cosine = (dir * direction);
			
			if (cosine < innerCosine) {
				if (cosine < outerCosine) continue;
				float a = ((innerCosine - cosine) / (innerCosine - outerCosine));
				float b = (1.0f - a);
				col *= pow((a * b * b) + (b * (1.0f - (a * a))), outerFalloff);
			}
			col.a = 1.0f;
			result->add(Photon(Ray(pos, dir), col));
		}
	}


	inline bool Spotlight::fromDson(
		const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context) {
		const Dson::Dict *dict = object.safeConvert<Dson::Dict>(errorStream, "Error: Spotlight can only be constructed from a dict"); if (dict == NULL) return false;

		Color shade = Color(1, 1, 1);
		if (dict->contains("color")) {
			Vector3 col(0, 0, 0); if (!col.fromDson(dict->get("color"), errorStream)) return false;
			shade = (ColorRGB)col;
		}
		
		float lum = 1500;
		if (dict->contains("luminosity")) {
			const Dson::Number *number = dict->get("luminosity").safeConvert<Dson::Number>(errorStream, "Error: Spotlight luminosity must be a number"); if (number == NULL) return false;
			lum = number->floatValue();
		}

		Vertex pos = Vertex(0, 0, 0);
		if (dict->contains("position"))
			if (!pos.fromDson(dict->get("position"), errorStream)) return false;
		
		Vector3 dir = Vector3(0, 0, 1);
		if (dict->contains("target")) {
			Vector3 target; if (!target.fromDson(dict->get("target"), errorStream)) return false;
			dir = (target - pos);
		}
		else if (dict->contains("direction")) 
		if (!dir.fromDson(dict->get("direction"), errorStream)) return false;

		float innerAngle = 16;
		if (dict->contains("hotspot_angle")) {
			const Dson::Number *number = dict->get("hotspot_angle").safeConvert<Dson::Number>(errorStream, "Error: Spotlight hotspot angle must be a number"); if (number == NULL) return false;
			innerAngle = number->floatValue();
		}
		
		float outerAngle = 64;
		if (dict->contains("falloff_angle")) {
			const Dson::Number *number = dict->get("falloff_angle").safeConvert<Dson::Number>(errorStream, "Error: Spotlight falloff angle must be a number"); if (number == NULL) return false;
			outerAngle = number->floatValue();
		}

		float falloffPower = 1.0f;
		if (dict->contains("falloff_power")) {
			const Dson::Number *number = dict->get("falloff_power").safeConvert<Dson::Number>(errorStream, "Error: Spotlight falloff power must be a number"); if (number == NULL) return false;
			falloffPower = number->floatValue();
		}
		
		float discSize = 0.25f;
		if (dict->contains("emitter_size")) {
			const Dson::Number *number = dict->get("emitter_size").safeConvert<Dson::Number>(errorStream, "Error: Spotlight emitter size must be a number"); if (number == NULL) return false;
			discSize = number->floatValue();
		}

		bool castShadows = true;
		if (dict->contains("cast_shadows")) {
			const Dson::Bool *value = dict->get("cast_shadows").safeConvert<Dson::Bool>(errorStream, "Error: Spotlight shadow cast flag must be a boolean"); if (value == NULL) return false;
			castShadows = value->value();
		}
		
		int samples = 1;
		if (dict->contains("sample_count")) {
			const Dson::Number *number = dict->get("sample_count").safeConvert<Dson::Number>(errorStream, "Error: Spotlight sample count must be a number"); if (number == NULL) return false;
			samples = number->intValue();
		}

		new(this)Spotlight(shade, lum, pos, dir, innerAngle, outerAngle, falloffPower, discSize, castShadows, samples);
		return true;
	}
}


