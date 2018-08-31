#pragma once
#include "../Lense.cuh"


namespace Lenses {
	class SphericalSegmentLense {
	public:
		__dumb__ SphericalSegmentLense(Vector2 angle = Vector2(60.0f, 60.0f), float focalDistance = 128.0f, float softness = 1.0f, Color sensitivity = Color(1.0f, 1.0f, 1.0f, 1.0f));

		__dumb__ void getPixelSamples(const LenseGetPixelSamplesRequest &request, RaySamples *samples)const;
		__dumb__ Color getPixelColor(const LenseGetPixelColorRequest &request)const;

		inline bool fromDson(const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *);

	private:
		Vector2 alpha;
		float dist, soft;
		Color filter;
	};
}




#include "SphericalSegmentLense.impl.cuh"
