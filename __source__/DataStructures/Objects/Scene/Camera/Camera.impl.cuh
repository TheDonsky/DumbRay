#include"Camera.cuh"





__dumb__ void Camera::getPhotons(const Vector2 &screenSpacePosition, PhotonPack &result)const {
	int i = result.size();
	lense.getScreenPhoton(screenSpacePosition, result);
	while (i < result.size()) {
		result[i] >>= transform;
		i++;
	}
}
__dumb__ void Camera::getColor(const Vector2 &screenSpacePosition, Photon photon, Color &result)const {
	photon <<= transform;
	// __TODO__???
}








/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** CUDA communication: **/
IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(Camera);
TYPE_TOOLS_IMPLEMENT_2_PART(Camera);
