#include"SimpleDirectionalLight.cuh"





__dumb__ SimpleDirectionalLight::SimpleDirectionalLight(Photon photon) {
	this->photon = photon;
}
__dumb__ void SimpleDirectionalLight::getPhotons(const Vertex &targetPoint, bool *noShadows, PhotonPack &result)const {
	(*noShadows) = false;
	result.push(Photon(Ray(targetPoint - photon.ray.direction * 1024.0f, photon.ray.direction), photon.color));
}
__dumb__ ColorRGB SimpleDirectionalLight::ambient(const Vertex &)const {
	return ColorRGB(0.0f, 0.0f, 0.0f);
}



