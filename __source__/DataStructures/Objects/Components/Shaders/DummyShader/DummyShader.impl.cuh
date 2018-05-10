#include"DummyShader.cuh"


__dumb__ void DummyShader::requestIndirectSamples(const ShaderInirectSamplesRequest<BakedTriFace> &, RaySamples *)const { 
#ifdef __CUDA_ARCH__
	if (threadIdx.x == 0 && blockIdx.x == 0)
		printf("From DEVICE, I inform you that I'm a duumy shader and won't do any good to you.\n");
#else
	printf("From HOST, I inform you that I'm a duumy shader and won't do any good to you.\n");
#endif
}
__dumb__ Color DummyShader::getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &)const { return Color(); }

