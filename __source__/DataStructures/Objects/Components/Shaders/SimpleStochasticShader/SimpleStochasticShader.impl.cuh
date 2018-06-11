#include "SimpleStochasticShader.cuh"


__dumb__ SimpleStochasticShader::SimpleStochasticShader(const ColorRGB &color, float diffuse, float smoothness, float shininess) {
	albedo = color;
	register float totalReflect = diffuse + smoothness;
	register float mul = ((totalReflect > 1.0f) ? (1.0f / totalReflect) : 1.0f);
	diff = diffuse * mul;
	gloss = smoothness * mul;
	shine = shininess;
}

__dumb__ void SimpleStochasticShader::requestIndirectSamples(const ShaderIndirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const {
	DumbRand &drand = (*request.context->entropy);
	register Vector3 direction;
	drand.pointOnSphere(direction.x, direction.y, direction.z);
	if ((direction * request.object->vert.normal()) < 0) direction = -direction;

	register Vector3 hitMasses = request.object->vert.getMasses(request.hitPoint);
	register Vector3 hitNormal = request.object->norm.massCenter(hitMasses);
	register Vector3 reflection = request.ray.direction.reflection(hitNormal);

	samples->set(SampleRay(Ray(request.hitPoint, ((direction * diff) + (reflection * gloss)).normalized()), 1.0f));
}
__dumb__ Color SimpleStochasticShader::getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const {
	register Vector3 hitMasses = request.object->vert.getMasses(request.hitPoint);
	register Vector3 hitNormal = request.object->norm.massCenter(hitMasses);
	register Vector3 reflection = request.photon.ray.direction.reflection(hitNormal);
	
	register float reflectionCos = reflection.angleCos(hitNormal);
	if (reflectionCos < 0.0f) reflectionCos = 0.0f;
	
	register float observerCos = request.observerDirection.angleCos(reflection);
	if (observerCos < 0.0f) observerCos = 0.0f;

	return ((request.photon.color * albedo) * ((diff * reflectionCos) + (gloss * pow(observerCos, shine))));
}


inline bool SimpleStochasticShader::fromDson(const Dson::Object &object, std::ostream *errorStream) {
	const Dson::Dict *dict = object.safeConvert<Dson::Dict>(errorStream, "Error: SimpleStochasticShader can not be constructed from any other Dson::Object but Dson::Dict...");
	if (dict == NULL) return false;
	Vector3 color = albedo;
	float diffuse = diff;
	float smoothness = gloss;
	float shininess = shine;
	if (dict->contains("color"))
		if (!color.fromDson(dict->get("color"), errorStream)) return false;
	if (dict->contains("diffuse")) {
		const Dson::Number *number = dict->get("diffuse").safeConvert<Dson::Number>(errorStream, "Error: SimpleStochasticShader diffuse has to be a number.");
		if (number == NULL) return false;
		diffuse = number->floatValue();
	}
	if (dict->contains("smoothness")) {
		const Dson::Number *number = dict->get("smoothness").safeConvert<Dson::Number>(errorStream, "Error: SimpleStochasticShader smoothness has to be a number.");
		if (number == NULL) return false;
		smoothness = number->floatValue();
	}
	if (dict->contains("shininess")) {
		const Dson::Number *number = dict->get("shininess").safeConvert<Dson::Number>(errorStream, "Error: SimpleStochasticShader shininess has to be a number.");
		if (number == NULL) return false;
		shininess = number->floatValue();
	}
	(*this) = SimpleStochasticShader(color, diffuse, smoothness, shininess);
	return true;
}
