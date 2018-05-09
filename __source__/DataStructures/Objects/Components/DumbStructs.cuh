#pragma once
#include"../../Primitives/Compound/Photon/Photon.cuh"
#include"../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include"../../GeneralPurpose/DumbRand/DumbRand.cuh"



typedef Stacktor<Photon, 16> PhotonPack;

template<typename Type>
struct DumbSamples {
	Type samples[16];
	int sampleCount;

	__device__ __host__ inline void add(const Type &sample) {
		samples[sampleCount] = sample;
		sampleCount++;
	}

	template<typename... Rest>
	__device__ __host__ inline void add(const Type &sample, const Rest&... rest) {
		add(sample);
		add(rest...);
	}

	template<typename... Elems>
	__device__ __host__ inline void set(const Elems&... elems) {
		sampleCount = 0;
		add(elems...);
	}
};

struct SampleRay {
	Ray ray;
	float sampleWeight;

	__device__ __host__ inline SampleRay() {}
	__device__ __host__ inline SampleRay(Ray sample, float mass) {
		ray = sample; sampleWeight = mass;
	}
};

typedef DumbSamples<Photon> PhotonSamples;
typedef DumbSamples<SampleRay> RaySamples;

enum PhotonType {
	PHOTON_TYPE_DIRECT_ILLUMINATION,	// Set, if the photon was directly cast from the light source.
	PHOTON_TYPE_INDIRECT_ILLUMINATION	// Set, if the photon is "back-casted" from the previously requested sample.
};

struct RenderContext {
	DumbRand *entropy;
};
