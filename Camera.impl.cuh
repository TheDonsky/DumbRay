#include"Camera.cuh"





__dumb__ Photon Camera::getPhoton(const Vector2 &screenSpacePosition)const {
	return (lense.getScreenPhoton(screenSpacePosition) >> transform);
}








/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** CUDA communication: **/
IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(Camera);
TYPE_TOOLS_IMPLEMENT_2_PART(Camera);
