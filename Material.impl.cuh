#include"Material.cuh"





namespace MaterialPrivateKernels {
	template<typename Shader, typename HitType>
	__global__ void getCastFn(typename Material<HitType>::CastFunction *destination) {
		(*destination) = Material<HitType>::template castOnShader<Shader>;
	}

	template<typename Shader, typename HitType>
	inline static typename Material<HitType>::CastFunction getDeviceCastFunction() {
		cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return NULL;
		else {
			typename Material<HitType>::CastFunction rv = NULL;
			typename Material<HitType>::CastFunction *fnPtr;
			if (cudaMalloc(&fnPtr, sizeof(Material<HitType>::CastFunction)) == cudaSuccess) {
				getCastFn<Shader, HitType> << <1, 1, 0, stream >> >(fnPtr);
				if (cudaStreamSynchronize(stream) == cudaSuccess) {
					bool success = (cudaMemcpyAsync(&rv, fnPtr, sizeof(Material<HitType>::CastFunction), cudaMemcpyDeviceToHost, stream) == cudaSuccess);
					if (success) if (cudaStreamSynchronize(stream) != cudaSuccess) rv = NULL;
				}
				cudaFree(fnPtr);
			}
			if (cudaStreamDestroy(stream) != cudaSuccess) rv = NULL;
			return rv;
		}
	}
}





template<typename HitType>
__host__ inline void Material<HitType>::init(){
	hostShader = NULL;
	devShader = NULL;
	ownsOnHost = false;

	hostCast = NULL;
	devCast = NULL;

	disposeOnHost = NULL;
	disposeOnDevice = NULL;
}
template<typename HitType>
template<typename Shader, typename... Args>
__host__ inline bool Material<HitType>::init(const Args&... args) {
	Shader *shaderHos = new Shader(args...);
	if (shaderHos == NULL) return false;
	if (!init<Shader>(shaderHos)) delete shaderHos;
	else ownsOnHost = true;
	return ownsOnHost;
}
template<typename HitType>
template<typename Shader>
__host__ inline bool Material<HitType>::init(Shader *shader) {
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
	hostCast = Material::castOnShader<Shader>;
	devCast = deviceCastFunction;
	return true;
}
template<typename HitType>
__host__ inline bool Material<HitType>::dispose(){
	if (!disposeOnDevice(devShader)) return false;
	if (ownsOnHost) if (!disposeOnHost(hostShader)) return false;
	init();
	return true;
}

template<typename HitType>
__dumb__ Material<HitType>::ShaderReport Material<HitType>::cast(const HitType &object, const HitInfo &info) const {
#ifdef __CUDA_ARCH__
	return devCast(devShader, object, info);
#else
	return hostCast(hostShader, object, info);
#endif
}





template<typename HitType>
template<typename Shader>
__dumb__ Material<HitType>::ShaderReport Material<HitType>::castOnShader(void *shader, const HitType &object, const HitInfo &info) {
	return ((Shader*)shader)->cast(object, info);
}


template<typename HitType>
template<typename Shader>
__host__ bool Material<HitType>::disposeFnHost(void *&shader){
	if (shader == NULL) return true;
	Shader *shad = ((Shader*)shader);
	delete shad;
	shader = NULL;
	return true;
}
template<typename HitType>
template<typename Shader>
__host__ bool Material<HitType>::disposeFnDevice(void *&shader){
	if (shader == NULL) return true;
	if (Shader::disposeOnDevice((Shader*)shader))
		if(cudaFree(shader) == cudaSuccess)
			shader = NULL;
	return(shader == NULL);
}

