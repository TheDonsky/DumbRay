#include "SimpleSoftDirectionalLight.cuh"



__dumb__ SimpleSoftDirectionalLight::SimpleSoftDirectionalLight(const Color shade, const Vector3 &direction, float distance, float softness) {
	color = shade;
	dir = direction.normalized();
	dist = distance;
	soft = softness;
}

__dumb__ void SimpleSoftDirectionalLight::getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const {
	(*castShadows) = true;
	DumbRand &drand = (*request.context->entropy);
	Vector3 offset = (Vector3(drand.range(-1.0f, 1.0f), drand.range(-1.0f, 1.0f), drand.range(-1.0f, 1.0f)).normalized() * soft);
	Vector3 origin = (request.point - (dir * dist) + offset);
	Vector3 direction = (request.point - origin);
	result->set(Photon(Ray(origin, direction), color));
}
