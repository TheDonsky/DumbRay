#include "DumbRenderContext.cuh"
#include "../../Namespaces/MeshReader/MeshReader.h"
#include "../Screen/FrameBuffer/BlockBasedFrameBuffer/BlockBasedFrameBuffer.cuh"
#include "../Objects/Components/Lenses/SimpleStochasticLense/SimpleStochasticLense.cuh"
#include "../../Namespaces/Images/Images.cuh"
#include "DumbRenderContextRegistry.cuh"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <mutex>

namespace {
	static std::mutex registryLock;
	static DumbRenderContextRegistry *registry = NULL;
	
	typedef std::unordered_map<std::string, DumbRenderContextRegistry::MaterialFromDsonFunction> MaterialParserMap;
	typedef std::unordered_map<std::string, DumbRenderContextRegistry::LightFromDsonFunction> LightParserMap;
	typedef std::unordered_map<std::string, DumbRenderContextRegistry::LenseFromDsonFunction> LenseParserMap;

	MaterialParserMap materialParsers;
	LightParserMap lightParsers;
	LenseParserMap lenseParsers;


	typedef std::unordered_map<std::string, PolyMesh> MeshDict;
	typedef std::unordered_map<std::string, MeshDict> ObjDict;

	struct DumbRenderContextData {
		std::string sourcePath;

		std::unordered_map<std::string, int> materials;
		std::unordered_map<std::string, int> textures;

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
}


void DumbRenderContextRegistry::registerMaterialType(
	const std::string &typeName, MaterialFromDsonFunction fromDsonFunction) {
	materialParsers[typeName] = fromDsonFunction;
}
void DumbRenderContextRegistry::registerLightType(
	const std::string &typeName, LightFromDsonFunction fromDsonFunction) {
	lightParsers[typeName] = fromDsonFunction;
}
void DumbRenderContextRegistry::registerLenseType(
	const std::string &typeName, LenseFromDsonFunction fromDsonFunction) {
	lenseParsers[typeName] = fromDsonFunction;
}


#define CONTEXT ((DumbRenderContextData*)data)->

DumbRenderContext::DumbRenderContext() {
	{
		std::lock_guard<std::mutex> guard(registryLock);
		if (registry == NULL) registry = new DumbRenderContextRegistry();
	}
	DumbRenderContextData *dataObject = new DumbRenderContextData();
	data = ((void*)dataObject);
	if (data == NULL) return;
	dataObject->threadConfiguration = Renderer::ThreadConfiguration(Renderer::ThreadConfiguration::ALL_BUT_GPU_THREADS, 2);
	dataObject->rendererSettings.boxingMode = DumbRenderer::BOXING_MODE_HEIGHT_BASED;
	dataObject->rendererSettings.maxBounces = 2;
	dataObject->rendererSettings.samplesPerPixelX = 1;
	dataObject->rendererSettings.samplesPerPixelY = 1;
	dataObject->rendererSettings.pixelsPerGPUThread = 1;
}
DumbRenderContext::~DumbRenderContext() {
	DumbRenderContextData *dataObject = ((DumbRenderContextData*)data);
	if (dataObject != NULL) { delete dataObject; data = NULL; }
}


bool DumbRenderContext::fromFile(const std::string &filename, std::ostream *errorStream) {
	std::ifstream stream;
	stream.open(filename.c_str());
	std::string string((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
	if (stream.fail()) {
		if (errorStream != NULL) (*errorStream) << ("Error: Could not read file: \"" + filename + "\"..") << std::endl;
		return false;
	}
	Dson::Object *object = Dson::parse(string, errorStream);
	const std::string &src = CONTEXT sourcePath;
	{
		size_t len = (filename.length());
		while ((len <= filename.length()) && (len > 0) && (filename[len - 1] != '/') && (filename[len - 1] != '\\')) len--;
		CONTEXT sourcePath = filename.substr(0, len);
	}
	bool rv;
	if (object != NULL) {
		rv = fromDson(object, errorStream);
		delete object;
	}
	else {
		if (errorStream != NULL) (*errorStream) << "Error: Could not parse file: \"" << filename << "\"" << std::endl;
		rv = false;
	}
	CONTEXT sourcePath = src;
	return rv;
}

bool DumbRenderContext::fromDson(const Dson::Object *object, std::ostream *errorStream) {
	if (object == NULL) {
		if (errorStream != NULL) (*errorStream) << "Error: Render context can not be constructed from a NULL Dson::Object" << std::endl;
		return false;
	}
	if (object->type() != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Error: Render context can be constructed only from a Dson::Dict" << std::endl;
		return false;
	}
	const Dson::Dict &dict = (*((const Dson::Dict*)object));
	if (dict.contains("include")) {
		if (!includeFiles(dict.get("include"), errorStream)) return false;
	}
	if (dict.contains("materials")) {
		if (!parseMaterials(dict.get("materials"), errorStream)) return false;
	}
	if (dict.contains("lights")) {
		if (!parseLights(dict.get("lights"), errorStream)) return false;
	}
	if (dict.contains("objects")) {
		if (!parseObjects(dict.get("objects"), errorStream)) return false;
	}
	if (dict.contains("camera")) {
		if (!parseCamera(dict.get("camera"), errorStream)) return false;
	}
	if (CONTEXT camera.cpuHandle()->lense.object() == NULL)
		CONTEXT camera.cpuHandle()->lense.use<SimpleStochasticLense>(64.0f);
	if (dict.contains("renderer")) {
		if (!parseRenderer(dict.get("renderer"), errorStream)) return false;
	}
	return true;
}


bool DumbRenderContext::getImageId(const Dson::Object &object, int *imageId, std::ostream *errorStream) {
	if ((object.type() == Dson::Object::DSON_BOOL)
		|| (object.type() == Dson::Object::DSON_NULL)
		|| (object.type() == Dson::Object::DSON_NUMBER)) {
		(*imageId) = (-1);
	}
	else if (object.type() == Dson::Object::DSON_STRING) {
		const std::string &name = ((const Dson::String*)(&object))->value();
		std::unordered_map<std::string, int>::const_iterator it = CONTEXT textures.find("name::" + name);
		if (it != CONTEXT textures.end()) { (*imageId) = it->second; return true; }
		it = CONTEXT textures.find("png::" + name);
		if (it != CONTEXT textures.end()) { (*imageId) = it->second; return true; }

		if (errorStream != NULL) (*errorStream) << ("Error: Texture \"" + name + "\" not found") << std::endl;
		return false;
	}
	else if (object.type() == Dson::Object::DSON_DICT) {
		const Dson::Dict &dict = (*((const Dson::Dict*)(&object)));
		if (dict.contains("png")) {
			const Dson::String *fileNameObject = dict.get("png").safeConvert<Dson::String>(errorStream, "Error: Image 'png' entry MUST BE a string");
			if (fileNameObject == NULL) return false;
			const std::string &fileName = fileNameObject->value();
			std::string filePath = (CONTEXT sourcePath + fileName);
			{
				std::ifstream stream;
				stream.open(filePath);
				if (stream.fail()) filePath = fileName;
			}
			Texture texture;
			if (Images::getTexturePNG(texture, filePath) == Images::IMAGES_NO_ERROR) {
				(*imageId) = CONTEXT scene.textures.cpuHandle()->size();
				CONTEXT scene.textures.cpuHandle()->flush(1);
				CONTEXT scene.textures.cpuHandle()->operator[](*imageId).stealFrom(texture);
				CONTEXT textures["png::" + filePath] = (*imageId);
				CONTEXT textures["png::" + fileName] = (*imageId);
			}
			else {
				if (errorStream != NULL) (*errorStream) << ("Error: Could not read file: \"" + filePath + "\"") << std::endl;
				return false;
			}
		}
		// MAYBE... ADD OPTIONS TO ADD SOME OTHER WAYS TO GENERATE IMAGES....
		else {
			if (errorStream != NULL) (*errorStream) << "Error: Image dict incomplete" << std::endl;
			return false;
		}

		if (dict.contains("filtering")) {
			const Dson::String *filterObject = dict.get("filtering").safeConvert<Dson::String>(errorStream, "Error: Image 'filtering' entry MUST BE a string");
			if (filterObject == NULL) return false;
			const std::string &filter = filterObject->value();
			if (filter == "none") CONTEXT scene.textures.cpuHandle()->operator[](*imageId).setFiltering(Texture::FILTER_NONE);
			else if (filter == "bilinear") CONTEXT scene.textures.cpuHandle()->operator[](*imageId).setFiltering(Texture::FILTER_BILINEAR);
			else {
				if (errorStream != NULL) (*errorStream) << ("Error: Image filter can be only \"none\"/[\"bilinear\"] (got: \"" + filter + "\")") << std::endl;
				return false;
			}
		}

		if (dict.contains("name")) {
			const Dson::String *nameObject = dict.get("png").safeConvert<Dson::String>(errorStream, "Error: Image name entry MUST BE a string");
			if (nameObject == NULL) return false;
			CONTEXT textures["name::" + nameObject->value()] = (*imageId);
		}
	}
	else {
		if (errorStream != NULL) (*errorStream) << "Error: Unsupported dson type for texture" << std::endl;
		return false;
	}
	return true;
}





bool DumbRenderContext::parseMaterials(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type() != Dson::Object::DSON_ARRAY) {
		if (errorStream != NULL) (*errorStream) << "Error: Materials should be contained in Dson::Array" << std::endl;
		return false;
	}
	const Dson::Array &arr = (*((Dson::Array*)(&object)));
	for (size_t i = 0; i < arr.size(); i++)
		if (!parseMaterial(arr[i], errorStream)) return false;
	return true;
}
bool DumbRenderContext::parseLights(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type() != Dson::Object::DSON_ARRAY) {
		if (errorStream != NULL) (*errorStream) << "Error: Lights should be contained in Dson::Array" << std::endl;
		return false;
	}
	const Dson::Array &arr = (*((Dson::Array*)(&object)));
	for (size_t i = 0; i < arr.size(); i++)
		if (!parseLight(arr[i], errorStream)) return false;
	return true;
}
bool DumbRenderContext::parseObjects(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type() != Dson::Object::DSON_ARRAY) {
		if (errorStream != NULL) (*errorStream) << "Error: Objects should be contained in Dson::Array" << std::endl;
		return false;
	}
	const Dson::Array &arr = (*((Dson::Array*)(&object)));
	for (size_t i = 0; i < arr.size(); i++)
		if (!parseObject(arr[i], errorStream)) return false;
	return true;
}
bool DumbRenderContext::parseCamera(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type() != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Error: Camera should be contained in Dson::Dict" << std::endl;
		return false;
	}
	const Dson::Dict &dict = (*((Dson::Dict*)(&object)));
	if (dict.contains("lense")) {
		const Dson::Object &lenseObject = dict.get("lense");
		if (lenseObject.type() != Dson::Object::DSON_DICT) {
			if (errorStream != NULL) (*errorStream) << "Error: Camera lense has to be a Dson::Dict" << std::endl;
			return false;
		}
		const Dson::Dict &lense = (*((Dson::Dict*)(&lenseObject)));
		if (!lense.contains("type")) {
			if (errorStream != NULL) (*errorStream) << "Error: Lense has to have a type" << std::endl;
			return false;
		}
		else {
			const Dson::Object &typeObject = lense.get("type");
			if (typeObject.type() != Dson::Object::DSON_STRING) {
				if (errorStream != NULL) (*errorStream) << "Error: Lense type has to be a string" << std::endl;
				return false;
			}
			const std::string &type = ((Dson::String*)(&typeObject))->value();
			LenseParserMap::const_iterator it = lenseParsers.find(type);
			if (it == lenseParsers.end()) {
				if (errorStream != NULL) (*errorStream) << "Error: Unknown lense type: \"" << type << "\"" << std::endl;
				return false;
			}
			Lense tmpLense;
			if (!it->second(tmpLense, lense, errorStream, this)) return false;
			CONTEXT camera.cpuHandle()->lense = tmpLense;
		}
	}
	if (dict.contains("transform")) {
		Transform transform;
		if (!transform.fromDson(dict["transform"], errorStream)) return false;
		CONTEXT camera.cpuHandle()->transform = transform;
	}
	return true;
}
bool DumbRenderContext::parseRenderer(const Dson::Object &object, std::ostream *errorStream) {
	const Dson::Dict *renderer = object.safeConvert<Dson::Dict>(errorStream, "Error: Renderer should be contained in Dson::Dict");
	if (renderer == NULL) return false;

	if (renderer->contains("resources")) {
		const Dson::Dict *resources = renderer->get("resources").safeConvert<Dson::Dict>(errorStream, "Error: Renderer Resources has to be a Dson::Dict type");
		if (resources == NULL) return false;
		if (resources->contains("cpu")) {
			const Dson::Object &cpuObject = resources->get("cpu");
			if (cpuObject.type() == Dson::Object::DSON_STRING) {
				const std::string &cpuText = ((const Dson::String*)(&cpuObject))->value();
				if (cpuText == "all") CONTEXT threadConfiguration.configureCPU(Renderer::ThreadConfiguration::ALL);
				else if (cpuText == "all_but_gpu") CONTEXT threadConfiguration.configureCPU(Renderer::ThreadConfiguration::ALL_BUT_GPU_THREADS);
				else if (cpuText == "all_but_one_per_gpu") CONTEXT threadConfiguration.configureCPU(Renderer::ThreadConfiguration::ALL_BUT_THREAD_PER_GPU);
				else if (cpuText == "none") CONTEXT threadConfiguration.configureCPU(Renderer::ThreadConfiguration::NONE);
				else if (cpuText == "one") CONTEXT threadConfiguration.configureCPU(Renderer::ThreadConfiguration::ONE);
				else {
					if (errorStream != NULL) (*errorStream) << "Error: Renderer Resources CPU can only be one of 'all'/['all_but_gpu']/'all_but_one_per_gpu'/'none'/'one'/any_number" << std::endl;
					return false;
				}
			}
			else if (cpuObject.type() == Dson::Object::DSON_NUMBER)
				CONTEXT threadConfiguration.configureCPU(((const Dson::Number*)(&cpuObject))->intValue());
			else {
				if (errorStream != NULL) (*errorStream) << "Error: Renderer Resources CPU can only be one of 'all'/['all_but_gpu']/'all_but_one_per_gpu'/'none'/'one'/any_number" << std::endl;
				return false;
			}
		}
		if (resources->contains("gpu")) {
			const Dson::Number *gpu = resources->get("gpu").safeConvert<Dson::Number>(errorStream, "Error: Renderer Resources GPU can only be a number");
			if (gpu == NULL) return false;
			CONTEXT threadConfiguration.configureEveryGPU(gpu->intValue());
		}
		for (int i = 0; i < CONTEXT threadConfiguration.numDevices(); i++) {
			std::stringstream stream;
			stream << "gpu_" << i;
			const std::string &gpuKey = stream.str();
			if (resources->contains(gpuKey)) {
				const Dson::Number *gpu = resources->get(gpuKey).safeConvert<Dson::Number>(errorStream, "Error: Renderer Resources " + gpuKey + " can only be a number");
				if (gpu == NULL) return false;
				CONTEXT threadConfiguration.configureGPU(i, gpu->intValue());
			}
		}
	}
	
	if (renderer->contains("blocks")) {
		const Dson::Dict *blocks = renderer->get("blocks").safeConvert<Dson::Dict>(errorStream, "Error: Renderer Blocks has to be a Dson::Dict type");
		if (blocks == NULL) return false;
		int blockCutPerCpuThread = CONTEXT blockConfiguration.blockCutPerCpuThread();
		int blockCutPerGpuSM = CONTEXT blockConfiguration.blockCutPerGpuSM();
		bool forceDeviceInstanceUpdate = CONTEXT blockConfiguration.forceDeviceInstanceUpdate();
		if (blocks->contains("cpu_cut")) {
			const Dson::Number *cpuCut = blocks->get("cpu_cut").safeConvert<Dson::Number>(errorStream, "Error: Renderer Blocks CPU_CUT has to be a number");
			if (cpuCut == NULL) return false;
			blockCutPerCpuThread = ((cpuCut->intValue() >= 1) ? cpuCut->intValue() : 1);
		}
		if (blocks->contains("gpu_cut")) {
			const Dson::Number *gpuCut = blocks->get("gpu_cut").safeConvert<Dson::Number>(errorStream, "Error: Renderer Blocks GPU_CUT has to be a number");
			if (gpuCut == NULL) return false;
			blockCutPerGpuSM = ((gpuCut->intValue() >= 1) ? gpuCut->intValue() : 1);
		}
		if (blocks->contains("force_host_block_synchronisation")) {
			const Dson::Bool *forceSynch = blocks->get("force_host_block_synchronisation").safeConvert<Dson::Bool>(errorStream, "Error: Renderer Blocks force_host_block_synchronisation has to be a boolean");
			if (forceSynch == NULL) return false;
			forceDeviceInstanceUpdate = forceSynch->value();
		}
		CONTEXT blockConfiguration = BlockRenderer::BlockConfiguration(blockCutPerCpuThread, blockCutPerGpuSM, forceDeviceInstanceUpdate);
	}

	if (renderer->contains("pixel")) {
		const Dson::Dict *pixel = renderer->get("pixel").safeConvert<Dson::Dict>(errorStream, "Error: Renderer Pixel has to be a Dson::Dict type");
		if (pixel == NULL) return false;
		if (pixel->contains("boxing")) {
			const Dson::String *boxing = pixel->get("boxing").safeConvert<Dson::String>(errorStream, "Error: Renderer Pixel boxing type has to be a string");
			if (boxing == NULL) return false;
			const std::string &boxingText = boxing->value();
			if (boxingText == "height") CONTEXT rendererSettings.boxingMode = DumbRenderer::BOXING_MODE_HEIGHT_BASED;
			else if (boxingText == "width") CONTEXT rendererSettings.boxingMode = DumbRenderer::BOXING_MODE_WIDTH_BASED;
			else if (boxingText == "min") CONTEXT rendererSettings.boxingMode = DumbRenderer::BOXING_MODE_MIN_BASED;
			else if (boxingText == "max") CONTEXT rendererSettings.boxingMode = DumbRenderer::BOXING_MODE_MAX_BASED;
			else {
				if (errorStream != NULL) (*errorStream) << "Error: Renderer Pixel boxing mode can only be one of ['height']/'width'/'min'/'max'" << std::endl;
				return false;
			}
		}
		if (pixel->contains("bounces")) {
			const Dson::Number *bounces = pixel->get("bounces").safeConvert<Dson::Number>(errorStream, "Error: Renderer Pixel bounces has to be a number");
			if (bounces == NULL) return false;
			CONTEXT rendererSettings.maxBounces = bounces->intValue();
		}
		if (pixel->contains("samples_per_pixel")) {
			const Dson::Array *bounces = pixel->get("samples_per_pixel").safeConvert<Dson::Array>(errorStream, "Error: Renderer Pixel samples_per_pixel has to be an array, containing two numbers");
			if (bounces == NULL) return false;
			if (bounces->size() != 2 || bounces->get(0).type() != Dson::Object::DSON_NUMBER || bounces->get(1).type() != Dson::Object::DSON_NUMBER) {
				if (errorStream != NULL) (*errorStream) << "Error: Renderer Pixel samples_per_pixel has to be an array, containing two numbers" << std::endl;
				return false;
			}
			CONTEXT rendererSettings.samplesPerPixelX = ((const Dson::Number*)(&bounces->get(0)))->intValue();
			CONTEXT rendererSettings.samplesPerPixelY = ((const Dson::Number*)(&bounces->get(1)))->intValue();
		}
		if (pixel->contains("pixels_per_gpu_thread")) {
			const Dson::Number *pixelsPerGpuThread = pixel->get("pixels_per_gpu_thread").safeConvert<Dson::Number>(errorStream, "Error: Renderer Pixel pixels_per_gpu_thread has to be a number");
			if (pixelsPerGpuThread == NULL) return false;
			CONTEXT rendererSettings.pixelsPerGPUThread = pixelsPerGpuThread->intValue();
		}
	}

	return true;
}
bool DumbRenderContext::includeFiles(const Dson::Object &object, std::ostream *errorStream) {
	const Dson::Array *include = object.safeConvert<Dson::Array>(errorStream, "Error: Include should be a list of strings");
	if (include == NULL) return false;
	for (size_t i = 0; i < include->size(); i++) {
		const Dson::String *text = include->get(i).safeConvert<Dson::String>(errorStream, "Error: Can not include non-string objects");
		if (text == NULL) std::cout << std::endl;
		const std::string &fileName = text->value();
		std::string filePath;
		{
			filePath = (CONTEXT sourcePath + fileName);
			std::ifstream stream;
			stream.open(filePath);
			if (stream.fail()) filePath = fileName;
		}
		if (!fromFile(filePath, errorStream)) return false;
	}
	return true;
}


bool DumbRenderContext::parseMaterial(const Dson::Object &object, std::ostream *errorStream, int *materialId) {
	if (object.type() != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Error: Material should be contained in Dson::Dict" << std::endl;
		return false;
	}
	const Dson::Dict &dict = (*((Dson::Dict*)(&object)));
	if (!dict.contains("type")) {
		if (errorStream != NULL) (*errorStream) << "Error: Material has to have a type" << std::endl;
		return false;
	}
	else {
		const Dson::Object &typeObject = dict.get("type");
		if (typeObject.type() != Dson::Object::DSON_STRING) {
			if (errorStream != NULL) (*errorStream) << "Error: Material type has to be a string" << std::endl;
			return false;
		}
		const std::string &type = ((Dson::String*)(&typeObject))->value();
		MaterialParserMap::const_iterator it = materialParsers.find(type);
		if (it == materialParsers.end()) {
			if (errorStream != NULL) (*errorStream) << "Error: Unknown material type: \"" << type << "\"" << std::endl;
			return false;
		}
		Material<BakedTriFace> material;
		if (!it->second(material, dict, errorStream, this)) return false;
		if (materialId != NULL) (*materialId) = CONTEXT scene.materials.cpuHandle()->size();
		CONTEXT scene.materials.cpuHandle()->push(material);
	}
	if (dict.contains("name")) {
		const Dson::Object &entry = dict.get("name");
		if (entry.type() == Dson::Object::DSON_STRING)
			CONTEXT materials[((Dson::String*)(&entry))->value()] = (CONTEXT scene.materials.cpuHandle()->size() - 1);
	}
	return true;
}
bool DumbRenderContext::parseLight(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type() != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Error: Light should be contained in Dson::Dict" << std::endl;
		return false;
	}
	const Dson::Dict &dict = (*((Dson::Dict*)(&object)));
	if (!dict.contains("type")) {
		if (errorStream != NULL) (*errorStream) << "Error: Light has to have a type" << std::endl;
		return false;
	}
	else {
		const Dson::Object &typeObject = dict.get("type");
		if (typeObject.type() != Dson::Object::DSON_STRING) {
			if (errorStream != NULL) (*errorStream) << "Error: Light type has to be a string" << std::endl;
			return false;
		}
		const std::string &type = ((Dson::String*)(&typeObject))->value();
		LightParserMap::const_iterator it = lightParsers.find(type);
		if (it == lightParsers.end()) {
			if (errorStream != NULL) (*errorStream) << "Error: Unknown light type: \"" << type << "\"" << std::endl;
			return false;
		}
		Light light;
		if (!it->second(light, dict, errorStream, this)) return false;
		CONTEXT scene.lights.cpuHandle()->push(light);
	}
	return true;
}
bool getObjMesh(void *data, const Dson::Dict &dict, BakedTriMesh &mesh, std::ostream *errorStream) {
	if (!dict.contains("obj")) {
		if (errorStream != NULL) (*errorStream) << "Error: 'obj' entry missing from Object." << std::endl;
		return false;
	}
	const Dson::Object &objObject = dict["obj"];
	if (objObject.type() != Dson::Object::DSON_STRING) {
		if (errorStream != NULL) (*errorStream) << "Error: Value 'obj' entry must be a string." << std::endl;
		return false;
	}
	const std::string &objFileName = ((Dson::String*)(&objObject))->value();
	if (CONTEXT objectFiles.find(objFileName) == CONTEXT objectFiles.end()) {
		Stacktor<PolyMesh> meshes;
		Stacktor<String> names;
		std::string objFilePath;
		{
			std::ifstream stream;
			const std::string relativePath = (CONTEXT sourcePath + objFileName);
			stream.open(relativePath);
			if (!stream.fail()) objFilePath = relativePath;
			else objFilePath = objFileName;
		}
		if (!MeshReader::readObj(meshes, names, objFilePath)) {
			if (errorStream != NULL) (*errorStream) << ("Error: Could not read file: '" + objFileName + "' (" + objFilePath + ")") << std::endl;
			return false;
		}
		if (meshes.size() != names.size()) {
			if (errorStream != NULL) (*errorStream) << ("Error: File '" + objFileName + "' does not have equal amount of names and meshes") << std::endl;
			return false;
		}
		CONTEXT objectFiles[objFileName] = MeshDict();
		MeshDict &polyMeshDict = CONTEXT objectFiles[objFileName];
		for (int i = 0; i < meshes.size(); i++)
			polyMeshDict[names[i] + 0] = meshes[i];
	}
	const MeshDict &meshDict = CONTEXT objectFiles[objFileName];
	if (dict.contains("object")) {
		const Dson::Object &objectObject = dict["object"];
		if (objectObject.type() != Dson::Object::DSON_STRING) {
			if (errorStream != NULL) (*errorStream) << "Error: Value 'object' entry must be a string." << std::endl;
			return false;
		}
		const std::string &objectName = ((Dson::String*)(&objectObject))->value();
		MeshDict::const_iterator it = meshDict.find(objectName);
		if (it == meshDict.end()) {
			if (errorStream != NULL) (*errorStream) << ("Error: Object '" + objectName + "' not found in file '" + objFileName + "'") << std::endl;
			return false;
		}
		const BakedTriMesh bakedMesh = it->second.bake();
		for (int i = 0; i < bakedMesh.size(); i++) mesh.push(bakedMesh[i]);
	}
	else for (MeshDict::const_iterator it = meshDict.begin(); it != meshDict.end(); it++) {
		const BakedTriMesh bakedMesh = it->second.bake();
		for (int i = 0; i < bakedMesh.size(); i++) mesh.push(bakedMesh[i]);
	}
	return true;
}

bool DumbRenderContext::parseObject(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type() != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Error: Object should be contained in Dson::Dict" << std::endl;
		return false;
	}
	const Dson::Dict &dict = (*((Dson::Dict*)(&object)));
	if (!dict.contains("material")) {
		if (errorStream != NULL) (*errorStream) << "Error: Object has to have a material" << std::endl;
		return false;
	}
	
	if (!dict.contains("mesh")) {
		if (errorStream != NULL) (*errorStream) << "Error: Object has to have a mesh" << std::endl;
		return false;
	}
	const Dson::Object &meshObject = dict["mesh"];
	if (meshObject.type() != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Error: Object 'mesh' entry has to be a dict" << std::endl;
		return false;
	}
	const Dson::Dict &meshDict = (*((Dson::Dict*)(&meshObject)));

	Transform transform;
	if (dict.contains("transform"))
		if (!transform.fromDson(dict["transform"], errorStream)) return false;

	int materialId;
	{
		const Dson::Object &material = dict["material"];
		if (material.type() == Dson::Object::DSON_STRING) {
			const std::string &name = ((Dson::String*)(&material))->value();
			std::unordered_map<std::string, int>::const_iterator it = CONTEXT materials.find(name);
			if (it == CONTEXT materials.end()) {
				if (errorStream != NULL) (*errorStream) << ("Error: Material not found: \"" + name + "\"") << std::endl;
				return false;
			}
			else materialId = it->second;
		}
		else if (!parseMaterial(material, errorStream, &materialId)) return false;
	}

	BakedTriMesh mesh;
	{
		if (meshDict.contains("obj")) {
			if (!getObjMesh(data, meshDict, mesh, errorStream)) return false;
		}
		else if (meshDict.contains("primitive")) {
			// create a primitive...
		}
		else {
			if (errorStream != NULL) (*errorStream) << "Error: Object mesh could not be parsed" << std::endl;
			return false;
		}
	}

	for (int i = 0; i < mesh.size(); i++)
		CONTEXT scene.geometry.cpuHandle()->push(DumbRenderer::SceneType::GeometryUnit(mesh[i] >> transform, materialId));
	
	return true;
}


namespace {
	class IterationObserver {
	private:
		const FrameBuffer *buffer;
		int lastWidth, lastHeight;
		volatile unsigned int iterationCount, lastIterationCount;
		clock_t startTime, lastTime;
		size_t lastCommentLength;

	public:
		inline IterationObserver(const FrameBuffer *frameBuffer) {
			buffer = frameBuffer;
			buffer->getSize(&lastWidth, &lastHeight);
			iterationCount = lastIterationCount = 0;
			startTime = lastTime = clock();
			lastCommentLength = 0;
		}

		inline void iterationComplete() {
			int width, height; buffer->getSize(&width, &height);
			if ((width != lastWidth) || (height != lastHeight)) {
				lastWidth = width;
				lastHeight = height;
				iterationCount = lastIterationCount = 0;
				std::cout << "\r"; for (size_t i = 0; i < lastCommentLength; i++) std::cout << " ";
			}
			iterationCount++;
			clock_t now = clock();
			clock_t delta = (now - lastTime);
			if (delta >= CLOCKS_PER_SEC) {
				std::cout << "\r"; for (size_t i = 0; i < lastCommentLength; i++) std::cout << " ";
				std::stringstream stream;
				long long seconds = ((long long)(((double)(now - startTime)) / CLOCKS_PER_SEC));
				long long minutes = (seconds / 60); seconds = (seconds - (minutes * 60));
				long long hours = (minutes / 60); minutes = (minutes - (hours * 60));
				stream << std::setprecision(4) << "Iterations:" << iterationCount
					<< " (elapsed: " << hours << ":" << minutes << ":" << seconds
					<< "; Avg Iter/Sec: " << (((double)iterationCount) / ((double)(now - startTime)) * CLOCKS_PER_SEC)
					<< "; Iter/Sec:" << (((double)(iterationCount - lastIterationCount)) / ((double)delta) * CLOCKS_PER_SEC) << ")";
				const std::string &text = stream.str();
				std::cout << "\r" << text;
				lastCommentLength = text.length();
				lastTime = now;
				lastIterationCount = iterationCount;
			}
		}

		inline static void iterationCompleteCallback(void *observer) {
			((IterationObserver*)observer)->iterationComplete();
		}
	};
}



void DumbRenderContext::runWindowRender() {
	CONTEXT scene.geometry.cpuHandle()->build();
	CONTEXT scene.geometry.makeDirty();

	std::cout << "_____________________________________________________________" << std::endl;
	std::cout << "RENDERING:" << std::endl;
	std::cout << "    ________________________________________" << std::endl;
	std::cout << "    CPU threads: " << CONTEXT threadConfiguration.numHostThreads() << std::endl;
	if (CONTEXT threadConfiguration.numDevices() > 0) {
		std::cout << "    GPU threads: ";
		if (CONTEXT threadConfiguration.numDevices() == 1) std::cout << CONTEXT threadConfiguration.numDeviceThreads(0) << " [" << Device::getDeviceName(0) << "]" << std::endl;
		else {
			std::cout << std::endl;
			for (int i = 0; i < CONTEXT threadConfiguration.numDevices(); i++)
				std::cout << "        GPU " << i << ": " << CONTEXT threadConfiguration.numDeviceThreads(i) << " [" << Device::getDeviceName(i) << "]" << std::endl;
		}
	}
	std::cout << "    Block cut per CPU thread: " << CONTEXT blockConfiguration.blockCutPerCpuThread() << std::endl;
	std::cout << "    Block cut per GPU SM:     " << CONTEXT blockConfiguration.blockCutPerGpuSM() << std::endl;
	std::cout << "    ________________________________________" << std::endl;
	std::cout << "    Geometry:    " << CONTEXT scene.geometry.cpuHandle()->getData().size() << " tris" << std::endl;
	std::cout << "    Node count:  " << CONTEXT scene.geometry.cpuHandle()->getNodeCount() << std::endl;
	std::cout << "    Materials:   " << CONTEXT scene.materials.cpuHandle()->size() << std::endl;
	std::cout << "    Lights:      " << CONTEXT scene.lights.cpuHandle()->size() << std::endl;
	std::cout << "    Bounces:     " << CONTEXT rendererSettings.maxBounces << std::endl;
	std::cout << "_____________________________________________________________" << std::endl;

	FrameBufferManager frameBuffer;
	frameBuffer.cpuHandle()->use<BlockBuffer>();

	DumbRenderer renderer(CONTEXT threadConfiguration, CONTEXT blockConfiguration,
		&frameBuffer, &CONTEXT scene, &CONTEXT camera,
		CONTEXT rendererSettings.boxingMode, CONTEXT rendererSettings.maxBounces,
		CONTEXT rendererSettings.samplesPerPixelX, CONTEXT rendererSettings.samplesPerPixelY,
		CONTEXT rendererSettings.pixelsPerGPUThread);

	int renderingDevice = 0;
	for (int i = 0; i < CONTEXT threadConfiguration.numDevices(); i++)
		if (CONTEXT threadConfiguration.numDeviceThreads(i) > 0) {
			renderingDevice = i;
			break;
		}
	BufferedWindow bufferedWindow(renderer.automaticallySynchesHostBlocks() ? 0 : BufferedWindow::SYNCH_FRAME_BUFFER_FROM_DEVICE, "Render Viewport", NULL, renderingDevice);

	BufferedRenderProcess process;
	process.setBuffer(&frameBuffer);
	process.setInfinateTargetIterations();
	process.setTargetResolutionToWindowSize();
	process.setTargetDisplayWindow(&bufferedWindow);
	process.setRenderer(&renderer);

	IterationObserver observer(frameBuffer.cpuHandle());
	process.setIterationCompletionCallback(IterationObserver::iterationCompleteCallback, &observer);

	process.start();
	while (!bufferedWindow.windowClosed()) std::this_thread::sleep_for(std::chrono::milliseconds(32));
	process.end();
	{
		std::cout << std::endl << "Enter a name ending with '.png' to save the image: ";
		std::string line; std::getline(std::cin, line);
		size_t leadInWhiteSpaces; for (leadInWhiteSpaces = 0; leadInWhiteSpaces < line.length(); leadInWhiteSpaces++) 
			if (!iswspace(line[leadInWhiteSpaces])) break;
		size_t leadOutWhiteSpaces; for (leadOutWhiteSpaces = 0; leadOutWhiteSpaces < line.length(); leadOutWhiteSpaces++) 
			if (!iswspace(line[line.length() - leadOutWhiteSpaces - 1])) break;
		if (leadInWhiteSpaces < line.length()) {
			std::string filename = line.substr(leadInWhiteSpaces, line.length() - leadInWhiteSpaces - leadOutWhiteSpaces);
			if (filename.length() >= 4) if (filename.substr(filename.length() - 4, 4) == ".png") {
				std::cout << "Saving...." << std::endl;
				if (Images::saveBufferPNG(*frameBuffer.cpuHandle(), filename) == Images::IMAGES_NO_ERROR)
					std::cout << ("Image saved at: '" + filename + "'...") << std::endl;
				else std::cout << ("Failed to save image at: '" + filename + "'...") << std::endl;
			}
		}
	}
}


void DumbRenderContext::test() {
	std::cout << "Enter scene file name: ";
	std::string filename;
	std::getline(std::cin, filename);
	testFile(filename);
}
void DumbRenderContext::testFile(const std::string &filename) {
	DumbRenderContext context;
	if (!context.fromFile(filename, &std::cout)) {
		std::string line;
		std::getline(std::cin, line);
	}
	context.runWindowRender();
}
