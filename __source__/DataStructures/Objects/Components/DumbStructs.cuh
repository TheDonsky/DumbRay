#pragma once
#include"../../Primitives/Compound/Photon/Photon.cuh"
#include"../../Primitives/Pure/Vector2/Vector2.h"
#include"../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include"../../GeneralPurpose/DumbRand/DumbRand.cuh"
#include"../Meshes/BakedTriMesh/BakedTriMesh.h"
#include"Texture/Texture.cuh"



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

	// Anything really.... This will reappear, once some object is hit (defaults to 0).
	float significance;

	// Anything really.... This will be transferred from ShaderIndirectSamplesRequest 
	// directly to ShaderReflectedColorRequest with indirect illumination photon (0 is reserved)
	uint32_t type;

	__device__ __host__ inline SampleRay() {}
	__device__ __host__ inline SampleRay(Ray sample, float mass, float significance = 1.0f, uint32_t typeFlags = 0) {
		ray = sample; sampleWeight = mass; this->significance = significance; type = typeFlags;
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
	const Stacktor<Texture> *textures;
};


class DumbRenderContext;

struct ColoredTexture {
	Color color;
	int textureId;
	Vector2 tiling;
	Vector2 offset;

	__device__ __host__ inline ColoredTexture() {}
	__device__ __host__ inline ColoredTexture(Color col, int id, 
		const Vector2 &tile = Vector2(1.0f, 1.0f), const Vector2 &off = Vector2(0.0f, 0.0f)) { 
		color = col; 
		textureId = id;
		tiling = tile;
		offset = off;
	}
	__device__ __host__ inline ColoredTexture(Color col) : ColoredTexture(col, -1) {}
	__device__ __host__ inline ColoredTexture(int id, const Vector2 &tile = Vector2(1.0f, 1.0f), const Vector2 &off = Vector2(0.0f, 0.0f)) 
		: ColoredTexture(Color(1.0f, 1.0f, 1.0f, 1.0f), id, tile, off) {}

	__device__ __host__ inline Color operator()(const Vector2 &pos, const RenderContext *context)const { 
		if (textureId < 0) return color;
		Vector2 coord = ((pos ^ tiling) + offset);
		return (color * context->textures->operator[](textureId)(coord));
	}

	bool fromDson(
		const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context,
		const std::string &colorKey = "color", const std::string &textureKey = "texture",
		const std::string & tilingKey = "tiling", const std::string & offsetKey = "offset");

	__device__ __host__ inline Vector3 getNormal(const Vector3 n, const Vector3 u, const Vector3 v, const Vector2 &pos, const RenderContext *context)const {
		if (textureId < 0) return n;
		Vector2 coord = ((pos ^ tiling) + offset);
		ColorRGB rgb = context->textures->operator[](textureId)(coord);
		Vector3 masses((color.g * rgb.b), (color.r * ((rgb.r * 2.0f) - 1.0f)), (color.b * ((rgb.g * 2.0f) - 1.0f)));
		return ((n * masses.x) + (u * masses.y) + (v * masses.z)).normalized();
	}
	__device__ __host__ inline Vector3 getNormal(const Vector3 n, const Vector3 u, const Vector2 &pos, const RenderContext *context)const {
		return getNormal(n, u, (u & n), pos, context);
	}
	__device__ __host__ inline Vector3 getNormal(const BakedTriFace &face, const Vector3 &masses, const RenderContext *context)const {
		Vector3 n = face.norm.massCenter(masses).normalized();
		const Vector3 pos = face.tex.massCenter(masses);
		if (textureId >= 0) {
			Vector2 texDelta;
			Vector3 verDelta;
			if (masses.x < 0.999f) {
				texDelta = (face.tex.a - pos);
				verDelta = (face.vert.a - face.vert.massCenter(masses));
			}
			else {
				texDelta = (face.tex.b - pos);
				verDelta = (face.vert.b - face.vert.massCenter(masses));
			}
			Vector3 verCoDelta = (verDelta & n).normalized();
			verDelta = (verCoDelta & n);

			Vector3 u = ((verDelta * texDelta.y) - (verCoDelta * texDelta.x)).normalized();

			n = getNormal(n, u, pos, context);
		}
		return n;
	}
};

