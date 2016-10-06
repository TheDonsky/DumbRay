#pragma once

#include"Vector3.h"





class Material{
public:
	__host__ inline void init();
	template<typename Shader, typename... Args>
	__host__ inline bool init(const Args&... args);
	template<typename Shader>
	__host__ inline bool init(Shader *shader);
	__host__ inline bool dispose();





private:
	void *hostShader;
	void *devShader;
	bool ownsOnHost;

	bool(*disposeOnHost)(void*);
	bool(*disposeOnDevice)(void*);

	template<typename Shader>
	__host__ static bool disposeFnHost(void *&shader);
	template<typename Shader>
	__host__ static bool disposeFnDevice(void *&shader);
};




#include"Material.impl.cuh"
