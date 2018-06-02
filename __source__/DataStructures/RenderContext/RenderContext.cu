#include "RenderContext.cuh"


namespace {
	typedef std::unordered_map<std::string, RenderContext::MaterialFromDsonFunction> MaterialParserMap;
	static MaterialParserMap materialParsers;
	typedef std::unordered_map<std::string, RenderContext::LightFromDsonFunction> LightParserMap;
	static LightParserMap lightParsers;
	typedef std::unordered_map<std::string, RenderContext::LenseFromDsonFunction> LenseParserMap;
	static LenseParserMap lenseParsers;
}


RenderContext::RenderContext() {

}
RenderContext::~RenderContext() {

}

bool RenderContext::fromDson(const Dson::Object *object, std::ostream *errorStream) {
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
	if (!dict.contains("camera")) {
		if (!parseCamera(dict.get("camera"), errorStream)) return false;
	}
	else {
		if (errorStream != NULL) (*errorStream) << "Error: Scene Dson HAS TO contain camera" << std::endl;
		return false;
	}
	return true;
}

void RenderContext::registerMaterialType(
	const std::string &typeName, MaterialFromDsonFunction fromDsonFunction) {
	materialParsers[typeName] = fromDsonFunction;
}
void RenderContext::registerLightType(
	const std::string &typeName, LightFromDsonFunction fromDsonFunction) {
	lightParsers[typeName] = fromDsonFunction;
}
void RenderContext::registerLenseType(
	const std::string &typeName, LenseFromDsonFunction fromDsonFunction) {
	lenseParsers[typeName] = fromDsonFunction;
}





bool RenderContext::parseMaterials(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type != Dson::Object::DSON_ARRAY) {
		if (errorStream != NULL) (*errorStream) << "Error: Materials should be contained in Dson::Array" << std::endl;
		return false;
	}
	const Dson::Array &arr = (*((Dson::Array*)(&object)));
	for (size_t i = 0; i < arr.size(); i++) {
		if (!parseMaterial(arr[i], errorStream)) return false;
	}
	return true;
}
bool RenderContext::parseLights(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type != Dson::Object::DSON_ARRAY) {
		if (errorStream != NULL) (*errorStream) << "Error: Lights should be contained in Dson::Array" << std::endl;
		return false;
	}
	const Dson::Array &arr = (*((Dson::Array*)(&object)));
	for (size_t i = 0; i < arr.size(); i++) {
		if (!parseLight(arr[i], errorStream)) return false;
	}
	return true;
}
bool RenderContext::parseObjects(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type != Dson::Object::DSON_ARRAY) {
		if (errorStream != NULL) (*errorStream) << "Error: Objects should be contained in Dson::Array" << std::endl;
		return false;
	}
	const Dson::Array &arr = (*((Dson::Array*)(&object)));
	for (size_t i = 0; i < arr.size(); i++) {
		if (!parseObject(arr[i], errorStream)) return false;
	}
	return true;
}
bool RenderContext::parseCamera(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type != Dson::Object::DSON_DICT) {
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
			// Try parse... (type dependent)
		}
	}
	return true;
}


bool RenderContext::parseMaterial(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type != Dson::Object::DSON_DICT) {
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
		if (!it->second(material, dict, errorStream)) return false;
		scene.materials.cpuHandle()->push(material);
	}
	if (dict.contains("name")) {
		const Dson::Object &entry = dict.get("name");
		if (entry.type() == Dson::Object::DSON_STRING)
			materials[((Dson::String*)(&entry))->value()] = (scene.materials.cpuHandle()->size() - 1);
	}
	return true;
}
bool RenderContext::parseLight(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type != Dson::Object::DSON_DICT) {
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
		Light material;
		if (!it->second(material, dict, errorStream)) return false;
		scene.lights.cpuHandle()->push(material);
	}
	return true;
}
bool RenderContext::parseObject(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Error: Object should be contained in Dson::Dict" << std::endl;
		return false;
	}
	const Dson::Dict &dict = (*((Dson::Dict*)(&object)));
	// Try parse..
	return true;
}

