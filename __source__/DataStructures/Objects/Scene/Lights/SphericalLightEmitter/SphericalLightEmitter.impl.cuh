#include "SphericalLightEmitter.cuh"



__dumb__ SphericalLightEmitter::SphericalLightEmitter(const Color shade, const Vector3 &position, float radius) {
	col = shade;
	pos = position;
	rad = radius;
}
__dumb__ void SphericalLightEmitter::getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const {
	(*castShadows) = true;
	Vector3 emissionPoint;
	request.context->entropy->pointOnSphere(emissionPoint.x, emissionPoint.y, emissionPoint.z, rad);
	emissionPoint *= request.context->entropy->getFloat();
	emissionPoint += pos;
	Vector3 delta = (request.point - emissionPoint);
	result->set(Photon(Ray(emissionPoint, delta.normalized()), col / delta.sqrMagnitude()));
}


inline bool SphericalLightEmitter::fromDson(const Dson::Object &object, std::ostream *errorStream) {
	const Dson::Dict *dict = object.safeConvert<Dson::Dict>(errorStream, "Error: SphericalLightEmitter can not be constructed from any other Dson::Object but Dson::Dict...");
	if (dict == NULL) return false;
	Color shade = col;
	Vector3 position = pos;
	float radius = rad;
	if (dict->contains("color")) {
		Vector3 color;
		if (!color.fromDson(dict->get("color"), errorStream)) return false;
		shade = ((ColorRGB)color);
	}
	if (dict->contains("position"))
		if (!position.fromDson(dict->get("position"), errorStream)) return false;
	if (dict->contains("radius")) {
		const Dson::Number *number = dict->get("radius").safeConvert<Dson::Number>(errorStream, "Error: SphericalLightEmitter radius has to be a number.");
		if (number == NULL) return false;
		radius = number->floatValue();
	}
	(*this) = SphericalLightEmitter(shade, position, radius);
	return true;
}
