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

