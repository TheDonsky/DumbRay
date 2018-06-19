#include "Texture.cuh"


__device__ __host__ inline Texture::Texture(uint32_t width, uint32_t height, Filtering filtering) {
	data = NULL;
	w = h = 0;
	protectedData = NULL;
	setReolution(width, height);
	setFiltering(filtering);
}
__device__ __host__ inline Texture::Texture(const Texture &other) : Texture() { copyFrom(other); }
__device__ __host__ inline Texture& Texture::operator=(const Texture &other) {
	copyFrom(other);
	return (*this);
}
__device__ __host__ inline void Texture::copyFrom(const Texture &other) {
	if (this == (&other)) return;
	setReolution(other.w, other.h);
	if (data != NULL) {
		register uint32_t surface = (w * h);
		for (uint32_t i = 0; i < surface; i++) data[i] = other.data[i];
	}
	setFiltering(other.flags);
}
__device__ __host__ inline Texture::Texture(Texture &&other) : Texture() { swapWith(other); }
__device__ __host__ inline Texture& Texture::operator=(Texture &&other) {
	swapWith(other);
	return (*this);
}
__device__ __host__ inline void Texture::stealFrom(Texture &other) {
	if (this == (&other)) return;
	clean();
	swapWith(other);
}
__device__ __host__ inline void Texture::swapWith(Texture &other) {
	if (this == (&other)) return;
	if ((protectedData != NULL) && (other.protectedData != NULL)) {
		TypeTools<Color*>::swap(data, other.data);
		TypeTools<uint32_t>::swap(w, other.w);
		TypeTools<uint32_t>::swap(h, other.h);
		TypeTools<Filtering>::swap(flags, other.flags);
	}
	else {
		Texture tmp = other;
		other.copyFrom(*this);
		copyFrom(tmp);
	}
}
__device__ __host__ inline Texture::~Texture() { clean(); }

__device__ __host__ inline void Texture::setReolution(uint32_t width, uint32_t height) {
	const uint32_t surface = (width * height);
	if (surface > (w * h)) {
		clean();
		if (surface > 0) data = new Color[surface];
	}
	if (data != NULL) {
		w = width;
		h = height;
	}
	else w = h = 0;
}
__device__ __host__ inline void Texture::setFiltering(Filtering filter) { flags = filter; }
__device__ __host__ inline void Texture::clean() {
	if (data != protectedData && data != NULL) delete[] data;
	data = NULL;
	w = h = 0;
}

__device__ __host__ inline uint32_t Texture::width()const { return w; }
__device__ __host__ inline uint32_t Texture::height()const { return h; }
__device__ __host__ inline Texture::Filtering Texture::filtering()const { return flags; }
__device__ __host__ inline Color* Texture::operator[](uint32_t y) { return (data + (y * w)); }
__device__ __host__ inline const Color* Texture::operator[](uint32_t y)const { return (data + (y * w)); }
__device__ __host__ inline Color& Texture::operator()(uint32_t x, uint32_t y) { return operator[](y)[x]; }
__device__ __host__ inline const Color& Texture::operator()(uint32_t x, uint32_t y)const { return operator[](y)[x]; }
__device__ __host__ inline const Color Texture::operator()(const Vector2 &pos)const {
	register float x = (pos.x + ((float)(((int)(-pos.x)) + 2))); x -= ((float)((int)x));
	register float y = (pos.y + ((float)(((int)(-pos.y)) + 2))); y -= ((float)((int)y)); y = (1.0f - y);
	register float posX = (x * ((float)(w - 1))), posY = (y * ((float)(h - 1)));
	register uint32_t minX = ((uint32_t)posX), minY = ((uint32_t)posY);
	register uint32_t maxX = ((minX + 1) % w), maxY = ((minY + 1) % h);
	register float offX = (posX - ((float)minX)), offY = (posY - ((float)minY));
	// __TMP__:
	return operator()((offX <= 0.5f) ? minX : maxX, (offY <= 0.5f) ? minY : maxY);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(Texture);





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<>
__device__ __host__ inline void TypeTools<Texture>::init(Texture &m) { new (&m) Texture(); }
template<>
__device__ __host__ inline void TypeTools<Texture>::dispose(Texture &m) { m.~Texture(); }
template<>
__device__ __host__ inline void TypeTools<Texture>::swap(Texture &a, Texture &b) { a.swapWith(b); }
template<>
__device__ __host__ inline void TypeTools<Texture>::transfer(Texture &src, Texture &dst) { dst.stealFrom(src); }

template<>
inline bool TypeTools<Texture>::prepareForCpyLoad(const Texture *source, Texture *hosClone, Texture *devTarget, int count) {
	cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
	int i = 0;
	for (i = 0; i < count; i++) {
		const Texture &texture = source[i];
		Texture &clone = hosClone[i];
		size_t bytes = (((size_t)(texture.w * texture.h)) * sizeof(Color));
		if (bytes == 0) clone.data = clone.protectedData = NULL;
		else if (cudaMalloc((void**)&clone.data, bytes) != cudaSuccess) break;
		else if (cudaMemcpyAsync((void*)clone.data, (void*)texture.data, bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) { cudaFree(clone.data); break; }
		else if (cudaStreamSynchronize(stream) != cudaSuccess) { cudaFree(clone.data); break; }
		else clone.protectedData = clone.data;
		clone.w = texture.w;
		clone.h = texture.h;
		clone.flags = texture.flags;
	}
	bool success = (i >= count);
	if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
	if (!success)
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
	return success;
}
template<>
inline void TypeTools<Texture>::undoCpyLoadPreparations(const Texture *, Texture *hosClone, Texture *, int count) {
	for (int i = 0; i < count; i++) if (hosClone[i].protectedData != NULL) cudaFree(hosClone[i].protectedData);
}
template<>
inline bool TypeTools<Texture>::devArrayNeedsToBeDisposed() { return true; }

namespace {
	namespace TexturePrivateKernels {
		__global__ static void disposeTextures(Texture *arr, int count) {
			int index = ((blockIdx.x * blockDim.x) + threadIdx.x);
			if (index < count) arr[index].~Texture();
		}
	}
}

template<>
inline bool TypeTools<Texture>::disposeDevArray(Texture *arr, int count) {
	if (count <= 0) return true;
	
	cudaStream_t kernelStream; if (cudaStreamCreate(&kernelStream) != cudaSuccess) return false;
	cudaStream_t memcpyStream; if (cudaStreamCreate(&memcpyStream) != cudaSuccess) { cudaStreamDestroy(kernelStream); return false; }
	
	TexturePrivateKernels::disposeTextures<<<(count + 255) / 256, 256, 0, kernelStream>>>(arr, count);

	bool success = true;
	size_t bytes = (sizeof(Texture) * count);
	void *hostCloneAddress;
	char hostGarbage[sizeof(Texture)];
	char *hostCloneBuffer;
	if (count > 0) {
		hostCloneBuffer = new char[bytes];
		hostCloneAddress = hostCloneBuffer;
	}
	else {
		hostCloneBuffer = NULL;
		hostCloneAddress = hostGarbage;
	}

	if (hostCloneAddress != NULL) {
		success = (cudaMemcpyAsync(hostCloneAddress, (void*)arr, bytes, cudaMemcpyDeviceToHost, memcpyStream) == cudaSuccess);
		if (cudaStreamSynchronize(memcpyStream) != cudaSuccess) success = false;
	}
	else success = false;
	if (cudaStreamDestroy(memcpyStream) != cudaSuccess) success = false;

	if (success) {
		Texture *hosClone = ((Texture*)hostCloneAddress);
		for (int i = 0; i < count; i++)
			if (hosClone[i].protectedData != NULL)
				if (cudaFree(hosClone[i].protectedData) != cudaSuccess) success = false;
	}

	if (cudaStreamSynchronize(kernelStream) != cudaSuccess) success = false;
	if (cudaStreamDestroy(kernelStream) != cudaSuccess) success = false;
	
	if (hostCloneBuffer != NULL) delete[] hostCloneBuffer;

	return success;
}

