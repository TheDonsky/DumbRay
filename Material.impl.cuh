#include"Material.cuh"




__host__ inline void Material::init(){
	hostShader = NULL;
	devShader = NULL;
	ownsOnHost = false;
	disposeOnHost = NULL;
	disposeOnDevice = NULL;
}
template<typename Shader, typename... Args>
__host__ inline bool Material::init(const Args&... args){
	Shader *shaderHos = new Shader(args...);
	if (init<Shader>(shaderHos))
		ownsOnHost = true;
	return ownsOnHost;
}
template<typename Shader>
__host__ inline bool Material::init(Shader *shader){
	init();
	if (shader == NULL) return false;
	Shader *shaderDev; if (cudaMalloc(&shaderDev, sizeof(Shader)) != cudaSuccess) return false;
	if (!(shader->uploadAt(shaderDev))) return false;
	hostShader = ((void*)shader);
	devShader = ((void*)shaderDev);
	disposeOnHost = disposeFnHost<Shader>;
	disposeOnDevice = disposeFnDevice<Shader>;
	return true;
}
__host__ inline bool Material::dispose(){
	if (!disposeOnDevice(devShader)) return false;
	if (ownsOnHost) if (!disposeOnHost(hostShader)) return false;
	init();
	return true;
}





template<typename Shader>
__host__ bool disposeFnHost(void *&shader){
	if (shader == NULL) return true;
	Shader *shad = ((Shader*)shader);
	delete shad;
	shader = NULL;
	return true;
}
template<typename Shader>
__host__ bool disposeFnDevice(void *&shader){
	if (shader == NULL) return true;
	if (Shader::disposeOnDevice((Shader*)shader))
		if(cudaFree(shader) == cudaSuccess)
			shader = NULL;
	return(shader == NULL);
}
