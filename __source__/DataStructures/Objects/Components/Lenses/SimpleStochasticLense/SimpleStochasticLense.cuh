#pragma once
#include "../Lense.cuh"


/*
This one functions identically to the DefaultPerspectiveLense,
but actually adds some entropy to the requested sample, 
so that multiple iterations look a lot like anti aliasing of some sorts...
*/

class SimpleStochasticLense {
public:
	__dumb__ SimpleStochasticLense(float angle = 60.0f);

	__dumb__ void getPixelSamples(const LenseGetPixelSamplesRequest &request, RaySamples *samples)const;
	__dumb__ Color getPixelColor(const LenseGetPixelColorRequest &request)const;

private:
	float x;
};


#include "SimpleStochasticLense.impl.cuh"
