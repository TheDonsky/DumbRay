#include"DummyShader.cuh"


__dumb__ ShaderReport DummyShader::cast(const ShaderHitInfo<BakedTriFace> &)const {
#ifdef __CUDA_ARCH__
	if (threadIdx.x == 0 && blockIdx.x == 0)
		printf("From DEVICE, I inform you that I'm a duumy shader and won't do any good to you.\n");
#else
	printf("From HOST, I inform you that I'm a duumy shader and won't do any good to you.\n");
#endif
	return ShaderReport();
}
__dumb__ void DummyShader::bounce(const ShaderBounceInfo<BakedTriFace> &, PhotonPack &)const { }
__dumb__ Photon DummyShader::illuminate(const ShaderHitInfo<BakedTriFace>&)const { return Photon(); }
