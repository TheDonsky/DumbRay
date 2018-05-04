#include"DefaultPerspectiveLense.cuh"



__dumb__ DefaultPerspectiveLense::DefaultPerspectiveLense(float angle) {
	if (angle < DEFAULT_PERSPECTIVE_LENSE_MIN_ANGLE) angle = DEFAULT_PERSPECTIVE_LENSE_MIN_ANGLE;
	else if (angle > DEFAULT_PERSPECTIVE_LENSE_MAX_ANGLE) angle = DEFAULT_PERSPECTIVE_LENSE_MAX_ANGLE;
	register float radian = (angle / (2 * RADIAN));
	x = 0.5f / tan(radian);
}

__dumb__ void DefaultPerspectiveLense::getPixelSamples(const Vector2 &screenSpacePosition, float, RaySamples *samples)const {
	samples->sampleCount = 1;
	samples->samples[0] = SampleRay(Ray(Vector3(0.0f, 0.0f, 0.0f), Vector3(screenSpacePosition.x, screenSpacePosition.y, x).normalized()), 1.0f);
}
__dumb__ Color DefaultPerspectiveLense::getPixelColor(const Vector2 &, const Photon &photon)const { return photon.color; }
