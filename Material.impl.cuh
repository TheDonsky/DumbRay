#include"Material.cuh"


namespace MaterialPrivateKernels {
	template<typename Shader, typename HitType>
	__global__ void getCastFn(Material::CastFunction *destination) {
		(*destination) = Material::cast<Shader, HitType>;
	}
	template<typename Shader, typename HitType>
	inline static Material::CastFunction getDeviceCastFunction() {
		cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return NULL;
		else {
			Material::CastFunction rv = NULL;
			Material::CastFunction *fnPtr;
			if (cudaMalloc(&fnPtr, sizeof(Material::CastFunction)) == cudaSuccess) {
				getCastFn<Shader, HitType><<<1, 1, 0, stream>>>(fnPtr);
				if (cudaStreamSynchronize(stream) == cudaSuccess) {
					bool success = (cudaMemcpyAsync(&rv, fnPtr, sizeof(Material::CastFunction), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
					if (success) if (cudaStreamSynchronize(stream) != cudaSuccess) rv = NULL;
				}
				cudaFree(fnPtr);
			}
			if (cudaStreamDestroy(stream) != cudaSuccess) rv = NULL;
			return rv;
		}
	}
}




__host__ inline void Material::init(){
	hostShader = NULL;
	devShader = NULL;
	ownsOnHost = false;

	hostCast = NULL;
	devCast = NULL;

	disposeOnHost = NULL;
	disposeOnDevice = NULL;
}
template<typename Shader, typename HitType, typename... Args>
__host__ inline bool Material::init(const Args&... args) {
	Shader *shaderHos = new Shader(args...);
	if (shaderHos == NULL) return false;
	if (!init<Shader, HitType>(shaderHos)) delete shaderHos;
	else ownsOnHost = true;
	return ownsOnHost;
}
template<typename Shader, typename HitType>
__host__ inline bool Material::init(Shader *shader) {
	init();
	if (shader == NULL) return false;
	Shader *shaderDev; if (cudaMalloc(&shaderDev, sizeof(Shader)) != cudaSuccess) return false;
	if (!(shader->uploadAt(shaderDev))){
		cudaFree(shaderDev);
		return false;
	}
	CastFunction deviceCastFunction = MaterialPrivateKernels::getDeviceCastFunction<Shader, HitType>();
	if (deviceCastFunction == NULL) {
		void *shaderDevice = (void*)shaderDev;
		if (!disposeFnDevice<Shader>(shaderDevice)) cudaFree(shaderDev);
		return false;
	}
	hostShader = ((void*)shader);
	devShader = ((void*)shaderDev);
	disposeOnHost = disposeFnHost<Shader>;
	disposeOnDevice = disposeFnDevice<Shader>;
	hostCast = Material::cast<Shader, HitType>;
	devCast = deviceCastFunction;
	return true;
}
__host__ inline bool Material::dispose(){
	if (!disposeOnDevice(devShader)) return false;
	if (ownsOnHost) if (!disposeOnHost(hostShader)) return false;
	init();
	return true;
}

__dumb__ Material::ShaderReport Material::cast(const HitInput &hit) const {
#ifdef __CUDA_ARCH__
	return devCast(devShader, hit);
#else
	return hostCast(hostShader, hit);
#endif
}





template<typename Shader, typename HitType>
__dumb__ Material::ShaderReport Material::cast(void *shader, const HitInput &hit) {
	ShaderInput<HitType> shaderInput = { *((HitType*)hit.object), hit.input };
	return ((Shader*)shader)->cast(shaderInput);
}


template<typename Shader>
__host__ bool Material::disposeFnHost(void *&shader){
	if (shader == NULL) return true;
	Shader *shad = ((Shader*)shader);
	delete shad;
	shader = NULL;
	return true;
}
template<typename Shader>
__host__ bool Material::disposeFnDevice(void *&shader){
	if (shader == NULL) return true;
	if (Shader::disposeOnDevice((Shader*)shader))
		if(cudaFree(shader) == cudaSuccess)
			shader = NULL;
	return(shader == NULL);
}
