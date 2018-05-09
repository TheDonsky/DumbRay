#include"SimpleDirectionalLight.cuh"





__dumb__ SimpleDirectionalLight::SimpleDirectionalLight(Photon photon) {
	this->photon = photon;
}

__dumb__ void SimpleDirectionalLight::getVertexPhotons(const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const {
	(*castShadows) = true;
	result->set(Photon(Ray(request.point - photon.ray.direction * 1024.0f, photon.ray.direction), photon.color));
}



