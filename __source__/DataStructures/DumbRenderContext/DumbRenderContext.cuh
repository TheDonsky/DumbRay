#pragma once
#include "../Renderers/DumbRenderer/DumbRenderer.cuh"
#include "../Renderers/BufferedRenderProcess/BufferedRenderProcess.cuh"
#include "../Objects/Meshes/PolyMesh/PolyMesh.h"
#include "../Objects/Scene/Scene.cuh"
#include "../../Namespaces/Dson/Dson.h"
#include <string>
#include <unordered_map>


class DumbRenderContext {
public:
	DumbRenderContext();
	~DumbRenderContext();

	bool fromFile(const std::string &filename, std::ostream *errorStream);
	bool fromDson(const Dson::Object *object, std::ostream *errorStream);


	typedef bool(*MaterialFromDsonFunction)(
		Material<BakedTriFace> &mat, 
		const Dson::Dict &object, std::ostream *errorStream);
	template <typename Type>
	inline static bool materialFromDson(
		Material<BakedTriFace> &mat, 
		const Dson::Dict &object, std::ostream *errorStream) {
		Type* materialObject = mat.use<Type>();
		if (materialObject == NULL) return false;
		else return materialObject->fromDson(object, errorStream);
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
		Type* lightObject = light.use<Type>();
		if (lightObject == NULL) return false;
		else return lightObject->fromDson(object, errorStream);
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
		Type* lenseObject = lense.use<Type>();
		if (lenseObject == NULL) return false;
		else return lenseObject->fromDson(object, errorStream);
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


	void runWindowRender();

	static void test();
	static void testFile(const std::string &filename);

private:
	__device__ __host__ inline DumbRenderContext(const DumbRenderContext &) {}
	__device__ __host__ inline DumbRenderContext& operator=(const DumbRenderContext &) { return (*this); }

	std::string sourcePath;

	bool parseMaterials(const Dson::Object &object, std::ostream *errorStream);
	bool parseLights(const Dson::Object &object, std::ostream *errorStream);
	bool parseObjects(const Dson::Object &object, std::ostream *errorStream);
	bool parseCamera(const Dson::Object &object, std::ostream *errorStream);
	bool parseRenderer(const Dson::Object &object, std::ostream *errorStream);
	bool includeFiles(const Dson::Object &object, std::ostream *errorStream);

	bool parseMaterial(const Dson::Object &object, std::ostream *errorStream, int *materialId = NULL);
	bool parseLight(const Dson::Object &object, std::ostream *errorStream);
	bool parseObject(const Dson::Object &object, std::ostream *errorStream);

	bool getObjMesh(const Dson::Dict &dict, BakedTriMesh &mesh, std::ostream *errorStream);

	std::unordered_map<std::string, int> materials;
	typedef std::unordered_map<std::string, PolyMesh> MeshDict;
	typedef std::unordered_map<std::string, MeshDict> ObjDict;
	ObjDict objectFiles;
	DumbRenderer::SceneType scene;
	ReferenceManager<Camera> camera;

	Renderer::ThreadConfiguration threadConfiguration;
	BlockRenderer::BlockConfiguration blockConfiguration;
	struct RendererSettings {
		DumbRenderer::BoxingMode boxingMode;
		int maxBounces;
		int samplesPerPixelX, samplesPerPixelY;
		int pixelsPerGPUThread;
	};
	RendererSettings rendererSettings;
};
