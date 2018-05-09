#include"SimpleDirectionalLight.cuh"





__dumb__ SimpleDirectionalLight::SimpleDirectionalLight(const Color shade, const Vector3 &direction, float distance) {
	color = shade;
	dir = direction.normalized();
	dist = distance;
}

__dumb__ void SimpleDirectionalLight::getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const {
	(*castShadows) = true;
	result->set(Photon(Ray(request.point - (dir * dist), dir), color));
}



