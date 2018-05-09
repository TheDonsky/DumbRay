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

	__dumb__ void getPixelSamples(const LenseGetPixelSamplesRequest &request, RaySamples *samples)const;
	__dumb__ Color getPixelColor(const LenseGetPixelColorRequest &request)const;

private:
	float x;
};






#include"DefaultPerspectiveLense.impl.cuh"
