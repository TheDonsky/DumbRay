#pragma once
#include "../../GeneralPurpose/ReferenceManager/ReferenceManager.cuh"
#include "../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include "Raycasters/Octree/Octree.cuh"
#include "Lights/Light.cuh"
#include "../Components/Shaders/Material.cuh"
#include "../Components/Texture/Texture.cuh"
#include "../Meshes/BakedTriMesh/BakedTriMesh.h"


template<typename HitType, typename GeometryType> struct SceneContext;

template<typename HitType, typename GeometryType>
struct Scene {
	typedef Material<HitType> MaterialType;
	typedef Stacktor<MaterialType> MaterialList;
	typedef ReferenceManager<MaterialList> MaterialManager;
	MaterialManager materials;

	typedef HitType SurfaceUnit;
	typedef Renderable<SurfaceUnit> GeometryUnit;
	typedef GeometryType Geometry;
	typedef ReferenceManager<Geometry> GeometryManager;
	GeometryManager geometry;
	
	typedef Stacktor<Light> LightList;
	typedef ReferenceManager<LightList> LightManager;
	LightManager lights;

	typedef Stacktor<Texture> TextureList;
	typedef ReferenceManager<TextureList> TextureManager;
	TextureManager textures;

	typedef SceneContext<HitType, GeometryType> Context;
};

template<typename HitType, typename GeometryType>
struct SceneContext {
	typedef Scene<HitType, GeometryType> SceneType;
	typename SceneType::MaterialList *materials;
	typename SceneType::Geometry *geometry;
	typename SceneType::LightList *lights;

	inline bool hasError() {
		return ((materials == NULL) || (geometry == NULL) || (lights == NULL));
	}
	inline void clean() {
		materials = NULL;
		geometry = NULL;
		lights = NULL;
	}
	inline bool host(SceneType *scene) {
		if (scene == NULL) { clean(); return false; }
		materials = scene->materials.cpuHandle();
		geometry = scene->geometry.cpuHandle();
		lights = scene->lights.cpuHandle();
		return (!hasError());
	}
	inline bool device(SceneType *scene, int deviceId) {
		if (scene == NULL) { clean(); return false; }
		materials = scene->materials.gpuHandle(deviceId);
		geometry = scene->geometry.gpuHandle(deviceId);
		lights = scene->lights.gpuHandle(deviceId);
		return (!hasError());
	}
};

typedef Scene<BakedTriFace, Octree<Renderable<BakedTriFace> > > TriScene;
