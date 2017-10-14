#include"Camera.cuh"





__dumb__ void Camera::getPhoton(const Vector2 &screenSpacePosition, PhotonPack &result)const {
	int i = result.size();
	lense.getScreenPhoton(screenSpacePosition, result);
	while (i < result.size()) {
		result[i] >>= transform;
		i++;
	}
}








/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** CUDA communication: **/
IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(Camera);
TYPE_TOOLS_IMPLEMENT_2_PART(Camera);
