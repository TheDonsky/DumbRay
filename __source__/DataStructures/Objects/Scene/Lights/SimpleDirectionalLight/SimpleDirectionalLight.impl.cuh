#include"SimpleDirectionalLight.cuh"





__dumb__ SimpleDirectionalLight::SimpleDirectionalLight(Photon photon) {
	this->photon = photon;
}

__dumb__ void SimpleDirectionalLight::getVertexPhotons(const Vector3 &point, PhotonSamples *result, bool *castShadows)const {
	(*castShadows) = true;
	result->set(Photon(Ray(point - photon.ray.direction * 1024.0f, photon.ray.direction), photon.color));
}



