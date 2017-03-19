#include"Camera.cuh"





__dumb__ Photon Camera::getPhoton(const Vector2 &screenSpacePosition)const {
	return (lense.getScreenPhoton(screenSpacePosition) >> transform);
}








/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** CUDA communication: **/
__dumb__ Transform &Camera::component0() {
	return transform;
}
__dumb__ const Transform &Camera::component0()const {
	return transform;
}
__dumb__ Lense &Camera::component1() {
	return lense;
}
__dumb__ const Lense &Camera::component1()const {
	return lense;
}
IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(Camera);
TYPE_TOOLS_IMPLEMENTATION_2(Camera);
