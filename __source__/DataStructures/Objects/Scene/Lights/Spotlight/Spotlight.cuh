#pragma once
#include"../Light.cuh"
#include "../../../../DumbRenderContext/DumbRenderContext.cuh"





namespace Lights {
	class Spotlight {
	public:
		__dumb__ Spotlight(
			Color shade = Color(1, 1, 1), float lum = 1500, 
			Vertex pos = Vertex(0, 0, 0), Vector3 dir = Vector3(0, 0, 1),
			float innerAngle = 16, float outerAngle = 64, float falloffPower = 1,
			float discSize = 0.25, bool castShadows = true, int samples = 1);
		__dumb__ void getVertexPhotons(
			const LightVertexSampleRequest &request, PhotonSamples *result, bool *castShadows)const;


		inline bool fromDson(
			const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context);
	
	private:
		Color color;
		Vertex position;
		Vector3 direction;
		float emitterSize;
		float innerCosine;
		float outerCosine;
		float outerFalloff;
		enum Flags {
			FLAGS_SAMPLE_COUNT_MASK = 31,
			FLAGS_CAST_SHADOWS = 32
		};
		uint8_t flags;
	};
}

#include "Spotlight.impl.cuh"