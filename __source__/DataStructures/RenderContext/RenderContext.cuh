#pragma once
#include "../Renderers/DumbRenderer/DumbRenderer.cuh"
#include "../Renderers/BufferedRenderProcess/BufferedRenderProcess.cuh"
#include "../Objects/Scene/Scene.cuh"
#include "../../Namespaces/Dson/Dson.h"


class Context {
public:
	Context();
	~Context();

	bool fromDson(Dson::Object *object);


private:
	__device__ __host__ inline Context(const Context &) {}
	__device__ __host__ inline Context& operator=(const Context &) {}

	TriScene scene;
	BufferedRenderProcess renderProcess;
};
