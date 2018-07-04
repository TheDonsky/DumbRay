#pragma once
#include "../Renderers/DumbRenderer/DumbRenderer.cuh"
#include "../Renderers/BufferedRenderProcess/BufferedRenderProcess.cuh"
#include "../Objects/Meshes/PolyMesh/PolyMesh.h"
#include "../Objects/Scene/Scene.cuh"
#include "DumbRenderContext.cuh"
#include <string>
#include <unordered_map>

class DumbRenderContextRegistry {
public:
	typedef bool(*MaterialFromDsonFunction)(
		Material<BakedTriFace> &mat,
		const Dson::Dict &object, std::ostream *errorStream, DumbRenderContext *context);
	typedef bool(*LightFromDsonFunction)(
		Light &light, const Dson::Dict &object, std::ostream *errorStream, DumbRenderContext *context);
	typedef bool(*LenseFromDsonFunction)(
		Lense &lense, const Dson::Dict &object, std::ostream *errorStream, DumbRenderContext *context);

private:
	template <typename Type>
	inline static bool materialFromDson(
		Material<BakedTriFace> &mat,
		const Dson::Dict &object, std::ostream *errorStream, DumbRenderContext *context) {
		Type* materialObject = mat.use<Type>();
		if (materialObject == NULL) return false;
		else return materialObject->fromDson(object, errorStream, context);
	}
	static void registerMaterialType(
		const std::string &typeName, MaterialFromDsonFunction fromDsonFunction);
	template <typename Type>
	inline static void registerMaterialType(const std::string &typeName) {
		registerMaterialType(typeName, materialFromDson<Type>);
	}

	template <typename Type>
	inline static bool lightFromDson(
		Light &light, const Dson::Dict &object, std::ostream *errorStream, DumbRenderContext *context) {
		Type* lightObject = light.use<Type>();
		if (lightObject == NULL) return false;
		else return lightObject->fromDson(object, errorStream, context);
	}
	static void registerLightType(
		const std::string &typeName, LightFromDsonFunction fromDsonFunction);
	template <typename Type>
	inline static void registerLightType(const std::string &typeName) {
		registerLightType(typeName, lightFromDson<Type>);
	}

	template <typename Type>
	inline static bool lenseFromDson(
		Lense &lense, const Dson::Dict &object, std::ostream *errorStream, DumbRenderContext *context) {
		Type* lenseObject = lense.use<Type>();
		if (lenseObject == NULL) return false;
		else return lenseObject->fromDson(object, errorStream, context);
	}
	static void registerLenseType(
		const std::string &typeName, LenseFromDsonFunction fromDsonFunction);
	template <typename Type>
	inline static void registerLenseType(const std::string &typeName) {
		registerLenseType(typeName, lenseFromDson<Type>);
	}

	static void registerMaterials();
	static void registerLights();
	static void registerLenses();
	
	DumbRenderContextRegistry() {
		registerMaterials();
		registerLights();
		registerLenses();
	}

	friend class DumbRenderContext;
};


