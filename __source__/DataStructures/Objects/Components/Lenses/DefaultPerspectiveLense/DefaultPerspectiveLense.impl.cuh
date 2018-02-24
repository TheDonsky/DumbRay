#include"DefaultPerspectiveLense.cuh"



__dumb__ DefaultPerspectiveLense::DefaultPerspectiveLense(float angle) {
	if (angle < DEFAULT_PERSPECTIVE_LENSE_MIN_ANGLE) angle = DEFAULT_PERSPECTIVE_LENSE_MIN_ANGLE;
	else if (angle > DEFAULT_PERSPECTIVE_LENSE_MAX_ANGLE) angle = DEFAULT_PERSPECTIVE_LENSE_MAX_ANGLE;
	register float radian = (angle / (2 * RADIAN));
	x = 0.5f / tan(radian);
}

__dumb__ void DefaultPerspectiveLense::getScreenPhoton(const Vector2 &screenSpacePosition, PhotonPack &result)const {
	result.push(Photon(
		Ray(Vector3(0.0f, 0.0f, 0.0f), 
			Vector3(screenSpacePosition.x, screenSpacePosition.y, x).normalized()), 
		ColorRGB(1.0f, 1.0f, 1.0f)));
}
__dumb__ Photon DefaultPerspectiveLense::toScreenSpace(const Photon &photon)const {
	register Vector3 delta = photon.ray.origin;
	register Vector3 direction = delta * (x / delta.y);
	return Photon(Ray(Vector3(0.0f, 0.0f, 0.0f), direction), photon.color);
}
__dumb__ void DefaultPerspectiveLense::getColor(const Vector2 &screenSpacePosition, const Photon &photon, Color &result)const {
	result = photon.color;
	// __TODO__???
}
