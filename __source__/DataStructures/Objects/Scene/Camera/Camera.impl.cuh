#include"Camera.cuh"





__dumb__ void Camera::getPixelSamples(const LenseGetPixelSamplesRequest &request, RaySamples *samples)const {
	lense.getPixelSamples(request, samples);
	for (int i = 0; i < samples->sampleCount; i++) samples->samples[i].ray >>= transform;
}
__dumb__ Color Camera::getPixelColor(LenseGetPixelColorRequest request)const {
	request.photon.ray <<= transform;
	return lense.getPixelColor(request);
}








/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** CUDA communication: **/
IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(Camera);
TYPE_TOOLS_IMPLEMENT_2_PART(Camera);
