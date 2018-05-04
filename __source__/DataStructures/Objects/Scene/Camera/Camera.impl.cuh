#include"Camera.cuh"





__dumb__ void Camera::getPixelSamples(const Vector2 &screenSpacePosition, float pixelSize, RaySamples &samples)const {
	lense.getPixelSamples(screenSpacePosition, pixelSize, &samples);
	for (int i = 0; i < samples.sampleCount; i++) samples.samples[i].ray >>= transform;
}
__dumb__ Color Camera::getPixelColor(const Vector2 &screenSpacePosition, const Photon &photon)const {
	return lense.getPixelColor(screenSpacePosition, Photon(photon.ray << transform, photon.color));
}








/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** CUDA communication: **/
IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(Camera);
TYPE_TOOLS_IMPLEMENT_2_PART(Camera);
