#pragma once
#include "../../Namespaces/Dson/Dson.h"
#include "DumbRenderContextConnector.cuh"


class DumbRenderContext {
public:
	DumbRenderContext();
	~DumbRenderContext();

	bool fromFile(const std::string &filename, std::ostream *errorStream);
	bool fromDson(const Dson::Object *object, std::ostream *errorStream);

	bool getImageId(const Dson::Object &object, int *imageId, std::ostream *errorStream);


	void runWindowRender();

	static void test();
	static void testFile(const std::string &filename);

private:
	inline DumbRenderContext(const DumbRenderContext &) {}
	inline DumbRenderContext& operator=(const DumbRenderContext &) { return (*this); }

	bool parseMaterials(const Dson::Object &object, std::ostream *errorStream);
	bool parseLights(const Dson::Object &object, std::ostream *errorStream);
	bool parseObjects(const Dson::Object &object, std::ostream *errorStream);
	bool parseCamera(const Dson::Object &object, std::ostream *errorStream);
	bool parseRenderer(const Dson::Object &object, std::ostream *errorStream);
	bool includeFiles(const Dson::Object &object, std::ostream *errorStream);

	bool parseMaterial(const Dson::Object &object, std::ostream *errorStream, int *materialId = NULL);
	bool parseLight(const Dson::Object &object, std::ostream *errorStream);
	bool parseObject(const Dson::Object &object, std::ostream *errorStream);


	void *data;
};
