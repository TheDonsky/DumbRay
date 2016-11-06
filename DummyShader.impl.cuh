#include"DummyShader.cuh"



inline bool DummyShader::uploadAt(DummyShader *dst) {
	return true;
}
inline bool DummyShader::disposeOnDevice(DummyShader *ptr) {
	return true;
}

__dumb__ Material<BakedTriFace>::ShaderReport DummyShader::cast(const Material<BakedTriFace>::HitInfo &input) {
#ifdef __CUDA_ARCH__
	if (threadIdx.x == 0 && blockIdx.x == 0)
		printf("From DEVICE, I inform you that I'm a duumy shader and won't do any good to you.\n");
#else
	printf("From HOST, I inform you that I'm a duumy shader and won't do any good to you.\n");
#endif
	return Material<BakedTriFace>::ShaderReport();
}
