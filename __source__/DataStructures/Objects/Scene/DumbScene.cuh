#pragma once
#include "../../GeneralPurpose/ReferenceManager/ReferenceManager.cuh"
#include "../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include "Raycasters/Octree/Octree.cuh"
#include "Lights/Light.cuh"
#include "../Components/Shaders/Material.cuh"
#include "../Meshes/BakedTriMesh/BakedTriMesh.h"


template<typename HitType, typename GeometryType> struct SceneContext;

template<typename HitType, typename GeometryType>
struct DumbScene {
	typedef Material<HitType> MaterialType;
	typedef Stacktor<MaterialType> MaterialList;
	typedef ReferenceManager<MaterialList> MaterialManager;
	MaterialManager materials;

	typedef Renderable<HitType> GeometryUnit;
	typedef GeometryType Geometry;
	typedef ReferenceManager<Geometry> GeometryManager;
	GeometryManager geometry;
	
	typedef Stacktor<Light> LightList;
	typedef ReferenceManager<LightList> LightManager;
	LightManager lights;

	typedef SceneContext<HitType, GeometryType> Context;
};

template<typename HitType, typename GeometryType>
struct SceneContext {
	typedef DumbScene<HitType, GeometryType> SceneType;
	SceneType::MaterialList *materials;
	SceneType::Geometry *geometry;
	SceneType::LightList *lights;

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

typedef DumbScene<BakedTriFace, Octree<Renderable<BakedTriFace> > > TriScene;
