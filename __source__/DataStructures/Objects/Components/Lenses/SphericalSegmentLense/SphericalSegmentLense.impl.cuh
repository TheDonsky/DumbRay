#include "SphericalSegmentLense.cuh"


namespace Lenses {
	
	__dumb__ SphericalSegmentLense::SphericalSegmentLense(Vector2 angle, float focalDistance, float softness, Color sensitivity) {
		alpha = angle;
		dist = focalDistance;
		soft = softness;
		filter = sensitivity;
	}

	__dumb__ void SphericalSegmentLense::getPixelSamples(const LenseGetPixelSamplesRequest &request, RaySamples *samples)const {
		Vector2 off = (request.pixelSize / 2.0f);
		DumbRand *dRand = request.context->entropy;
		
		register float horAngle = ((request.screenSpacePosition.x + dRand->range(-off.x, off.x)) * alpha.x / RADIAN);
		register float verAngle = ((request.screenSpacePosition.y + dRand->range(-off.y, off.y)) * alpha.y / RADIAN);

		register float horCos = cos(horAngle); register float horSin = sin(horAngle);

		register float verCos = cos(verAngle); register float verSin = sin(verAngle);

		register Vector3 targetDirection(horSin * verCos, verSin, horCos * verCos);

		register Vector3 target = (targetDirection * dist);

		register float randAngle = dRand->range(0.0f, 2.0f * PI);
		register Vector3 origin = (Vector3(cos(randAngle), sin(randAngle), 0.0f) * asin(dRand->getFloat()) * soft);

		samples->set(SampleRay(Ray(origin, (target - origin).normalize()), 1.0f));
	}
	__dumb__ Color SphericalSegmentLense::getPixelColor(const LenseGetPixelColorRequest &request)const {
		if (request.photonType == PHOTON_TYPE_DIRECT_ILLUMINATION) return Color(0.0f, 0.0f, 0.0f, 0.0f);
		else return (request.photon.color * filter);
	}

	inline bool SphericalSegmentLense::fromDson(const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *) {
		const Dson::Dict *dict = object.safeConvert<Dson::Dict>(errorStream, "SimpleFocalPlaneLense can not be built from non-dict object");
		if (dict == NULL) return false;

		if (dict->contains("angle")) {
			const Dson::Number *num = dict->get("angle").safeConvert<Dson::Number>(errorStream, "SimpleFocalPlaneLense angle must be a number");
			if (num == NULL) return false;
			alpha = Vector2(num->floatValue(), num->floatValue());
		}

		if (dict->contains("distance")) {
			const Dson::Number *num = dict->get("distance").safeConvert<Dson::Number>(errorStream, "SimpleFocalPlaneLense distance must be a number");
			if (num == NULL) return false;
			dist = num->floatValue();
		}

		if (dict->contains("sensor")) {
			const Dson::Number *num = dict->get("sensor").safeConvert<Dson::Number>(errorStream, "SimpleFocalPlaneLense sensor size must be a number");
			if (num == NULL) return false;
			soft = num->floatValue();
		}

		if (dict->contains("filter")) {
			Vector3 filterColor;
			if (!filterColor.fromDson(dict->get("filter"), errorStream)) return false;
			filter = (ColorRGB)filterColor;
		}

		return true;
	}
}
