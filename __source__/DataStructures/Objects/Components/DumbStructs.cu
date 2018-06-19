#include "DumbStructs.cuh"
#include "../../DumbRenderContext/DumbRenderContext.cuh"




bool ColoredTexture::fromDson(
	const Dson::Object &object, std::ostream *errorStream, DumbRenderContext *context,
	const std::string &colorKey, const std::string &textureKey,
	const std::string & tilingKey, const std::string & offsetKey) {
	
	const Dson::Dict *dict = object.safeConvert<Dson::Dict>(errorStream, "Error: Colored texture can only be constructed from a dict");
	if (dict == NULL) return false;

	if (dict->contains(colorKey)) {
		Vector3 colorVector(0.0f, 0.0f, 0.0f);
		if (!colorVector.fromDson(dict->get(colorKey), errorStream)) return false;
		color = (ColorRGB)colorVector;
	}

	if (dict->contains(textureKey))
		if (!context->getImageId(dict->get(textureKey), &textureId, errorStream)) return false;
	
	if (dict->contains(tilingKey)) {
		Vector3 tilingVector(0.0f, 0.0f, 0.0f);
		if (!tilingVector.fromDson(dict->get(tilingKey), errorStream)) return false;
		tiling = tilingVector;
	}

	if (dict->contains(offsetKey)) {
		Vector3 offsetVector(0.0f, 0.0f, 0.0f);
		if (!offsetVector.fromDson(dict->get(offsetKey), errorStream)) return false;
		offset = offsetVector;
	}

	return true;
}

