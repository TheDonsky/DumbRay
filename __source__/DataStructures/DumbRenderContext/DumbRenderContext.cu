#include "DumbRenderContext.cuh"
#include "../../Namespaces/MeshReader/MeshReader.h"
#include "../Screen/FrameBuffer/BlockBasedFrameBuffer/BlockBasedFrameBuffer.cuh"
#include <fstream>


namespace {
	typedef std::unordered_map<std::string, DumbRenderContext::MaterialFromDsonFunction> MaterialParserMap;
	typedef std::unordered_map<std::string, DumbRenderContext::LightFromDsonFunction> LightParserMap;
	typedef std::unordered_map<std::string, DumbRenderContext::LenseFromDsonFunction> LenseParserMap;
	
	class Registry {
	public:
		inline Registry() {
			DumbRenderContext::registerMaterials();
			DumbRenderContext::registerLights();
			DumbRenderContext::registerLenses();
		}

		MaterialParserMap materialParsers;
		LightParserMap lightParsers;
		LenseParserMap lenseParsers;
	};
	static Registry registry;
}


DumbRenderContext::DumbRenderContext() {

}
DumbRenderContext::~DumbRenderContext() {

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
	{
		size_t len = (filename.length());
		while ((len <= filename.length()) && (len > 0) && (filename[len - 1] != '/') && (filename[len - 1] != '\\')) len--;
		sourcePath = filename.substr(0, len);
	}
	bool rv = fromDson(object, errorStream);
	sourcePath = "";
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
	else {
		if (errorStream != NULL) (*errorStream) << "Error: Scene Dson HAS TO contain camera" << std::endl;
		return false;
	}
	return true;
}

void DumbRenderContext::registerMaterialType(
	const std::string &typeName, MaterialFromDsonFunction fromDsonFunction) {
	registry.materialParsers[typeName] = fromDsonFunction;
}
void DumbRenderContext::registerLightType(
	const std::string &typeName, LightFromDsonFunction fromDsonFunction) {
	registry.lightParsers[typeName] = fromDsonFunction;
}
void DumbRenderContext::registerLenseType(
	const std::string &typeName, LenseFromDsonFunction fromDsonFunction) {
	registry.lenseParsers[typeName] = fromDsonFunction;
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
	if (!dict.contains("lense")) {
		if (errorStream != NULL) (*errorStream) << "Error: Camera should have a lense" << std::endl;
		return false;
	}
	else {
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
			LenseParserMap::const_iterator it = registry.lenseParsers.find(type);
			if (it == registry.lenseParsers.end()) {
				if (errorStream != NULL) (*errorStream) << "Error: Unknown lense type: \"" << type << "\"" << std::endl;
				return false;
			}
			Lense tmpLense;
			if (!it->second(tmpLense, lense, errorStream)) return false;
			camera.cpuHandle()->lense = tmpLense;
		}
	}
	if (dict.contains("transform")) {
		Transform transform;
		if (!transform.fromDson(dict["transform"], errorStream)) return false;
		camera.cpuHandle()->transform = transform;
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
		MaterialParserMap::const_iterator it = registry.materialParsers.find(type);
		if (it == registry.materialParsers.end()) {
			if (errorStream != NULL) (*errorStream) << "Error: Unknown material type: \"" << type << "\"" << std::endl;
			return false;
		}
		Material<BakedTriFace> material;
		if (!it->second(material, dict, errorStream)) return false;
		if (materialId != NULL) (*materialId) = scene.materials.cpuHandle()->size();
		scene.materials.cpuHandle()->push(material);
	}
	if (dict.contains("name")) {
		const Dson::Object &entry = dict.get("name");
		if (entry.type() == Dson::Object::DSON_STRING)
			materials[((Dson::String*)(&entry))->value()] = (scene.materials.cpuHandle()->size() - 1);
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
		LightParserMap::const_iterator it = registry.lightParsers.find(type);
		if (it == registry.lightParsers.end()) {
			if (errorStream != NULL) (*errorStream) << "Error: Unknown light type: \"" << type << "\"" << std::endl;
			return false;
		}
		Light light;
		if (!it->second(light, dict, errorStream)) return false;
		scene.lights.cpuHandle()->push(light);
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
			std::unordered_map<std::string, int>::const_iterator it = materials.find(name);
			if (it == materials.end()) {
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
			if (!getObjMesh(meshDict, mesh, errorStream)) return false;
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
		scene.geometry.cpuHandle()->push(DumbRenderer::SceneType::GeometryUnit(mesh[i] >> transform, materialId));
	
	return true;
}

bool DumbRenderContext::getObjMesh(const Dson::Dict &dict, BakedTriMesh &mesh, std::ostream *errorStream) {
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
	if (objectFiles.find(objFileName) == objectFiles.end()) {
		Stacktor<PolyMesh> meshes;
		Stacktor<String> names;
		std::string objFilePath;
		{
			std::ifstream stream;
			stream.open(objFileName);
			if (!stream.fail()) objFilePath = objFileName;
			else objFilePath = (sourcePath + objFileName);
		}
		if (!MeshReader::readObj(meshes, names, objFilePath)) {
			if (errorStream != NULL) (*errorStream) << ("Error: Could not read file: '" + objFileName+ "' (" + objFilePath + ")") << std::endl;
			return false;
		}
		if (meshes.size() != names.size()) {
			if (errorStream != NULL) (*errorStream) << ("Error: File '" + objFileName + "' does not have equal amount of names and meshes") << std::endl;
			return false;
		}
		objectFiles[objFileName] = MeshDict();
		MeshDict &polyMeshDict = objectFiles[objFileName];
		for (int i = 0; i < meshes.size(); i++)
			polyMeshDict[names[i] + 0] = meshes[i];
	}
	const MeshDict &meshDict = objectFiles[objFileName];
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



void DumbRenderContext::runWindowRender() {
	scene.geometry.cpuHandle()->build();
	scene.geometry.makeDirty();
	
	Renderer::ThreadConfiguration configuration;
	configuration.configureEveryGPU(2);
	configuration.configureCPU(Renderer::ThreadConfiguration::ALL_BUT_GPU_THREADS);

	DumbRenderer renderer(configuration);
	renderer.setScene(&scene);
	renderer.setCamera(&camera);

	FrameBufferManager frameBuffer;
	frameBuffer.cpuHandle()->use<BlockBuffer>();

	bool shouldSynchFromDevice = (!((configuration.numHostThreads() > 0) || (configuration.numActiveDevices() > 1)));
	BufferedWindow bufferedWindow(shouldSynchFromDevice ? BufferedWindow::SYNCH_FRAME_BUFFER_FROM_DEVICE : 0);

	BufferedRenderProcess process;
	process.setBuffer(&frameBuffer);
	process.setInfinateTargetIterations();
	process.setTargetResolutionToWindowSize();
	process.setTargetDisplayWindow(&bufferedWindow);
	process.setRenderer(&renderer);


	process.start();
	while (!bufferedWindow.windowClosed()) std::this_thread::sleep_for(std::chrono::milliseconds(32));
	process.end();
}


void DumbRenderContext::test() {
	std::cout << "Enter scene file name: ";
	std::string filename;
	std::getline(std::cin, filename);
	testFile(filename);
}
void DumbRenderContext::testFile(const std::string &filename) {
	DumbRenderContext context;
	if (!context.fromFile(filename, &std::cout)) return;
	context.runWindowRender();
}
