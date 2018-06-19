#include "SimpleSoftDirectionalLight.cuh"



__dumb__ SimpleSoftDirectionalLight::SimpleSoftDirectionalLight(const Color shade, const Vector3 &direction, float distance, float softness) {
	color = shade;
	dir = direction.normalized();
	dist = distance;
	soft = (dist * softness);
}

__dumb__ void SimpleSoftDirectionalLight::getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const {
	(*castShadows) = true;
	DumbRand &drand = (*request.context->entropy);
	Vector3 offset; drand.pointOnSphere(offset.x, offset.y, offset.z, soft * drand.getFloat());
	Vector3 origin = (request.point - (dir * dist) + offset);
	Vector3 direction = (request.point - origin);
	result->set(Photon(Ray(origin, direction), color));
}


inline bool SimpleSoftDirectionalLight::fromDson(const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *) {
	const Dson::Dict *dict = object.safeConvert<Dson::Dict>(errorStream, "Error: SimpleSoftDirectionalLight can only be constructed from Dson::Dict...");
	Color shade = color;
	Vector3 direction = dir;
	float distance = dist;
	float softness = (soft / ((distance == 0.0f) ? 0.0f : distance));
	if (dict->contains("color")) {
		Vector3 colorVector(0.0f, 0.0f, 0.0f);
		if (!colorVector.fromDson(dict->get("color"), errorStream)) return false;
		shade = ((ColorRGB)colorVector);
	}
	if (dict->contains("direction")) {
		direction = Vector3(0.0f, 0.0f, 0.0f);
		if (!direction.fromDson(dict->get("direction"), errorStream)) return false;
	}
	if (dict->contains("distance")) {
		const Dson::Number *distanceObject = dict->get("distance").safeConvert<Dson::Number>(errorStream, "Error: SimpleSoftDirectionalLight distance has to have a numeric value...");
		if (distanceObject == NULL) return false;
		distance = distanceObject->floatValue();
	}
	if (dict->contains("softness")) {
		const Dson::Number *softnessObject = dict->get("softness").safeConvert<Dson::Number>(errorStream, "Error: SimpleSoftDirectionalLight softness has to have a numeric value...");
		if (softnessObject == NULL) return false;
		softness = softnessObject->floatValue();
	}
	(*this) = SimpleSoftDirectionalLight(shade, direction, distance, softness);
	return true;
}
