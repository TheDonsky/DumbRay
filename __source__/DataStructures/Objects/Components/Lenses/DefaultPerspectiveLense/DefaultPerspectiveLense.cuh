#pragma once
#include "../Lense.cuh"


#define DEFAULT_PERSPECTIVE_LENSE_MIN_ANGLE 1.0f
#define DEFAULT_PERSPECTIVE_LENSE_MAX_ANGLE 179.0f

// NOTE: This is the default lense that "cares" only about angles 
//		and does not simulate anything remotely connected to an actual
//		physical lense and the matrix. Things will always appear in focus.


class DefaultPerspectiveLense {
public:
	__dumb__ DefaultPerspectiveLense(float angle = 60.0f);

	__dumb__ void getScreenPhoton(const Vector2 &screenSpacePosition, PhotonPack &result)const;
	__dumb__ Photon toScreenSpace(const Photon &photon)const;
	__dumb__ void getColor(const Vector2 &screenSpacePosition, const Photon &photon, Color &result)const;


private:
	float x;
};






#include"DefaultPerspectiveLense.impl.cuh"
