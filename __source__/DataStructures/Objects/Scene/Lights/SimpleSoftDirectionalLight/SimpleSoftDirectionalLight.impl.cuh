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
