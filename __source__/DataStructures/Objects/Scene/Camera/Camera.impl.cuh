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


__dumb__ void Camera::getPixelSamples(const Vector2 &screenSpacePosition, float pixelSize, RaySamples &samples)const {
	lense.getPixelSamples(screenSpacePosition, pixelSize, &samples);
	for (int i = 0; i < samples.sampleCount; i++) samples.samples[i].ray >>= transform;
}








/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** CUDA communication: **/
IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(Camera);
TYPE_TOOLS_IMPLEMENT_2_PART(Camera);
