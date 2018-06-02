#pragma once
#include "../Renderers/DumbRenderer/DumbRenderer.cuh"
#include "../Renderers/BufferedRenderProcess/BufferedRenderProcess.cuh"
#include "../Objects/Meshes/PolyMesh/PolyMesh.h"
#include "../Objects/Scene/Scene.cuh"
#include "../../Namespaces/Dson/Dson.h"
#include <string>
#include <unordered_map>


class RenderContext {
public:
	RenderContext();
	~RenderContext();

	bool fromDson(const Dson::Object *object, std::ostream *errorStream);


	typedef bool(*MaterialFromDsonFunction)(
		Material<BakedTriFace> &mat, 
		const Dson::Dict &object, std::ostream *errorStream);
	template <typename Type>
	inline static bool materialFromDson(
		Material<BakedTriFace> &mat, 
		const Dson::Dict &object, std::ostream *errorStream) {
		Type* object = mat.use<Type>();
		if (object == NULL) return false;
		else return object->fromDson(object, errorStream);
	}
	static void registerMaterialType(
		const std::string &typeName, MaterialFromDsonFunction fromDsonFunction);
	template <typename Type>
	inline static void registerMaterialType(const std::string &typeName) {
		registerMaterialType(typeName, materialFromDson<Type>);
	}

	typedef bool(*LightFromDsonFunction)(
		Light &light, const Dson::Dict &object, std::ostream *errorStream);
	template <typename Type>
	inline static bool lightFromDson(
		Light &light, const Dson::Dict &object, std::ostream *errorStream) {
		Type* object = light.use<Type>();
		if (object == NULL) return false;
		else return object->fromDson(object, errorStream);
	}
	static void registerLightType(
		const std::string &typeName, LightFromDsonFunction fromDsonFunction);
	template <typename Type>
	inline static void registerLightType(const std::string &typeName) {
		registerLightType(typeName, lightFromDson<Type>);
	}

	typedef bool(*LenseFromDsonFunction)(
		Lense &lense, const Dson::Dict &object, std::ostream *errorStream);
	template <typename Type>
	inline static bool lenseFromDson(
		Lense &lense, const Dson::Dict &object, std::ostream *errorStream) {
		Type* object = lense.use<Type>();
		if (object == NULL) return false;
		else return object->fromDson(object, errorStream);
	}
	static void registerLenseType(
		const std::string &typeName, LenseFromDsonFunction fromDsonFunction);
	template <typename Type>
	inline static void registerLenseType(const std::string &typeName) {
		registerLenseType(typeName, lenseFromDson<Type>);
	}




private:
	__device__ __host__ inline RenderContext(const RenderContext &) {}
	__device__ __host__ inline RenderContext& operator=(const RenderContext &) {}


	bool parseMaterials(const Dson::Object &object, std::ostream *errorStream);
	bool parseLights(const Dson::Object &object, std::ostream *errorStream);
	bool parseObjects(const Dson::Object &object, std::ostream *errorStream);
	bool parseCamera(const Dson::Object &object, std::ostream *errorStream);

	bool parseMaterial(const Dson::Object &object, std::ostream *errorStream);
	bool parseLight(const Dson::Object &object, std::ostream *errorStream);
	bool parseObject(const Dson::Object &object, std::ostream *errorStream);

	std::unordered_map<std::string, int> materials;
	std::unordered_map<std::string, PolyMesh> meshes;
	DumbRenderer::SceneType scene;
};
