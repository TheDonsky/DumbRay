#pragma once
#include "../Renderers/BufferedRenderer/BufferedRenderer.cuh"
#include "../Objects/Scene/DumbScene.cuh"
#include "../../Namespaces/Dson/Dson.h"


class Context {
public:
	Context();
	~Context();

	bool fromDson();


private:
	__device__ __host__ inline Context(const Context &) {}
	__device__ __host__ inline Context& operator=(const Context &) {}


};
