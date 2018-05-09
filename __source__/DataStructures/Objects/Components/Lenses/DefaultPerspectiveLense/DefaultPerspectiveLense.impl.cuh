#include"DefaultPerspectiveLense.cuh"



__dumb__ DefaultPerspectiveLense::DefaultPerspectiveLense(float angle) {
	if (angle < DEFAULT_PERSPECTIVE_LENSE_MIN_ANGLE) angle = DEFAULT_PERSPECTIVE_LENSE_MIN_ANGLE;
	else if (angle > DEFAULT_PERSPECTIVE_LENSE_MAX_ANGLE) angle = DEFAULT_PERSPECTIVE_LENSE_MAX_ANGLE;
	register float radian = (angle / (2 * RADIAN));
	x = 0.5f / tan(radian);
}

__dumb__ void DefaultPerspectiveLense::getPixelSamples(const LenseGetPixelSamplesRequest &request, RaySamples *samples)const {
	samples->sampleCount = 1;
	samples->samples[0] = SampleRay(Ray(Vector3(0.0f, 0.0f, 0.0f), Vector3(request.screenSpacePosition.x, request.screenSpacePosition.y, x).normalized()), 1.0f);
}
__dumb__ Color DefaultPerspectiveLense::getPixelColor(const LenseGetPixelColorRequest &request)const { 
	if (request.photonType == PHOTON_TYPE_DIRECT_ILLUMINATION) return Color(0.0f, 0.0f, 0.0f, 0.0f);
	else return request.photon.color; 
}
