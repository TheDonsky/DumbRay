#include"BackwardTracer.cuh"




namespace BackwardTracerPrivate {
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	struct Pixel {
		struct PixelColor {
			ColorRGB color;
			float depth;
		};
		PixelColor currentIteration;
		PixelColor lastIteration;
		ColorRGB background;
	};
	template<typename HitType>
	struct SceneDataHandles {
		const ShadedOctree<HitType> *world;
		const Stacktor<Light> *lights;
		const Camera *camera;
	};





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
#define BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_WIDTH 16
#define BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_HEIGHT 8
#define BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_SIZE (BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_WIDTH * BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_HEIGHT)
	__dumb__ int blockWidth() { return BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_WIDTH; }
	__dumb__ int blockHeight() { return BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_HEIGHT; }
	__dumb__ int threadsPerBlock() { return (blockWidth() * blockHeight()); }
	__dumb__ int blocksWidth(int width) { return ((width + blockWidth() - 1) / blockWidth()); }
	__dumb__ int blocksHeight(int height) { return ((height + blockHeight() - 1) / blockHeight()); }
	__dumb__ int blockCount(int width, int height) { return (blocksWidth(width) * blocksHeight(height)); }

	__dumb__ void getKernelDimensions(int width, int height, int &blocks, int &threads) {
		threads = threadsPerBlock();
		blocks = blockCount(width, height);
	}

	__device__ inline static bool getPixelId(int width, int height, int &x, int &y, int blockOffset = 0) {
		register int blockId = (blockIdx.x + blockOffset);
		register int blocksW = blocksWidth(width);
		register int lineId = (blockId / blocksW);
		register int columnId = (blockId - (lineId * blocksW));

		register int blockW = blockWidth();
		register int threadLine = (threadIdx.x / blockW);
		register int threadColumn = (threadIdx.x - (threadLine * blockW));

		register int blockH = blockHeight();
		x = columnId * blockW + threadColumn;
		y = lineId * blockH + threadLine;
		return (x < width && y < height);
	}

	__dumb__ Vector2 toScreenSpace(float x, float y, int width, int height) {
		return ((Vector2(x, ((float)height - y)) - (Vector2((float)width, (float)height) * 0.5f)) / ((float)height));
	}

	__global__ static void cleanImage(Matrix<Pixel> *matrix, Color color) {
		int x, y; if (!getPixelId(matrix->width(), matrix->height(), x, y)) return;
		matrix->operator()(y, x).background = color;
		matrix->operator()(y, x).lastIteration.color = color;
		matrix->operator()(y, x).lastIteration.depth = FLT_MAX;
	}
	template<typename ColorType>
	__global__ static void loadImage(const Matrix<Pixel> *source, Matrix<ColorType> *destination) {
		int x, y; if (!getPixelId(source->width(), source->height(), x, y)) return;
		if (x < destination->width() && y < destination->height()) {
			ColorRGB color = source->operator()(y, x).lastIteration.color;
			if (color.r < 0) color.r = 0;
			else if (color.r > 1) color.r = 1;
			if (color.g < 0) color.g = 0;
			else if (color.g > 1) color.g = 1;
			if (color.b < 0) color.b = 0;
			else if (color.b > 1) color.b = 1;
			destination->operator()(y, x) = color;
		}
	}


	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
	/** ########################################################################## **/
	/** Main logic:                                                                **/
	/** ########################################################################## **/
	template<typename HitType>
	struct PixelRenderFrame {
		Photon photon;
		typename ShadedOctree<HitType>::RaycastHit hit;
		ColorRGB color;
		PhotonPack bounce;
		int bounceId;
	};

	template<typename HitType, unsigned int MaxStackSize>
	struct PixelRenderProcess {
		const SceneDataHandles<HitType> *world;
		PixelRenderFrame<HitType> stack[MaxStackSize];
		PixelRenderFrame<HitType> *end;
		PixelRenderFrame<HitType> *ptr;
		Pixel::PixelColor output;
		bool renderComplete;

		bool shadeStarted;
		bool midShade;
		Photon savedPhoton;
		int lightId;

		int castsLeft;

		__device__ __host__ inline bool shade() {
			PixelRenderFrame<HitType> &frame = (*ptr);

			if (!shadeStarted) {
				if (castsLeft <= 0) {
					midShade = true;
					return true;
				}
				else {
					midShade = false;
					shadeStarted = true;
					castsLeft--;
				}
				if (frame.photon.dead()) return false;
				if (!world->world->cast(frame.photon.ray, frame.hit, false)) return false;

				frame.color(0.0f, 0.0f, 0.0f);

				// BOUNCE FROM INDIRECT ILLUMINATION:
				if ((ptr + 1) != end) {
					ShaderBounceInfo<HitType> bounceInfo = { &frame.hit.object->object, frame.photon, frame.hit.hitPoint };
					frame.hit.object->material->bounce(bounceInfo, frame.bounce);
				}
				else frame.bounce.clear();
				frame.bounceId = 0;
			}

			// COLOR FROM LIGHT SOURCES:
			const Stacktor<Light> &lights = (*world->lights);
			for (int i = lightId; i < lights.size(); i++) {
				bool noShadows;
				Photon p;
				if (midShade) {
					noShadows = false;
					p = savedPhoton;
				}
				else {
					noShadows = false;
					PhotonPack pack;
					lights[i].getPhotons(frame.hit.hitPoint, &noShadows, pack);
					p = pack[0];
					if (p.dead()) continue;
				}
				if (!noShadows) {
					if (castsLeft > 0) {
						typename ShadedOctree<HitType>::RaycastHit lightHit;
						if (world->world->cast(p.ray, lightHit, false)) {
							if ((frame.hit.hitPoint - lightHit.hitPoint).sqrMagnitude() <= 128.0f * VECTOR_EPSILON)
								noShadows = true;
						}
						else noShadows = true;
						midShade = false;
						castsLeft--;
					}
					else {
						midShade = true;
						savedPhoton = p;
						lightId = i;
						return true;
					}
				}
				if (noShadows) {
					ShaderHitInfo<HitType> castInfo = { &frame.hit.object->object, p, frame.hit.hitPoint, frame.photon.ray.origin };
					frame.color += frame.hit.object->material->illuminate(castInfo).color;
				}
			}
			frame.color *= frame.photon.color;

			shadeStarted = false;
			midShade = false;
			lightId = 0;
			return true;
		}

		__dumb__ bool iterate() {
			while (true) {
				if (midShade || (ptr->bounceId < ptr->bounce.size())) {
					if (!midShade) {
						Photon sample = ptr->bounce[ptr->bounceId];
						sample.ray.origin += sample.ray.direction * (128.0f * VECTOR_EPSILON);
						ptr->bounceId++;
						ptr++;
						ptr->photon = sample;
					}
					if (!shade()) {
						if (ptr != stack) ptr--;
						else {
							output.depth = -1;
							renderComplete = true;
							return true;
						}
					}
					else if (midShade) return false;
				}
				else if (ptr == stack) {
					output.color = ptr->color;
					output.depth = ptr->hit.hitDistance;
					renderComplete = true;
					return true;
				}
				else {
					ColorRGB col = ptr->color;
					ptr--;
					ptr->color += col;
				}
			}
		}

		__dumb__ void setup(const Photon &photon, const SceneDataHandles<HitType> &world) {
			this->world = (&world);
			end = (stack + MaxStackSize);
			ptr = stack;
			ptr->photon = photon;
			shadeStarted = false;
			midShade = true;
			lightId = 0;
			renderComplete = false;
			castsLeft = 0;
		}
	};

	template<typename HitType, unsigned int MaxStackSize>
	struct PixelRenderBlock {
		//PixelRenderProcess<HitType, MaxStackSize> pixels[BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_WIDTH][BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_HEIGHT];
		int numCompleted;
	};

	template<typename HitType, unsigned int MaxStackSize>
	struct BackwardTracerRenderProcess {
		Stacktor<PixelRenderBlock<HitType, MaxStackSize> > pixels;
		SceneDataHandles<HitType> world;
		struct BlockBank {
			Cutex cutex;
			int ind;
			int delta;
			int end;
		};
		bool renderMustGoOn;
	};
}
/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** ########################################################################## **/
/** TypeTools:                                                                 **/
/** ########################################################################## **/
template<typename HitType>
class TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> > {
public:
	typedef BackwardTracerPrivate::PixelRenderFrame<HitType> Frame;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(Frame);
};
template<typename HitType>
__device__ __host__ inline void TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::init(Frame &m) {
	TypeTools<typename ShadedOctree<HitType>::RaycastHit>::init(m.hit);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::dispose(Frame &m) {
	TypeTools<typename ShadedOctree<HitType>::RaycastHit>::dispose(m.hit);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::swap(Frame &a, Frame &b) {
	TypeTools<Photon>::swap(a.photon, b.photon);
	TypeTools<typename ShadedOctree<HitType>::RaycastHit>::swap(a.hit, b.hit);
	TypeTools<ColorRGB>::swap(a.color, b.color);
	TypeTools<PhotonPack>::swap(a.bounce, b.bounce);
	TypeTools<int>::swap(a.bounceId, b.bounceId);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::transfer(Frame &src, Frame &dst) {
	TypeTools<Photon>::transfer(src.photon, dst.photon);
	TypeTools<typename ShadedOctree<HitType>::RaycastHit>::transfer(src.hit, dst.hit);
	TypeTools<ColorRGB>::transfer(src.color, dst.color);
	TypeTools<PhotonPack>::transfer(src.bounce, dst.bounce);
	TypeTools<int>::transfer(src.bounceId, dst.bounceId);
}
template<typename HitType>
inline bool TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::prepareForCpyLoad(const Frame *source, Frame *hosClone, Frame *devTarget, int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if (!TypeTools<typename ShadedOctree<HitType>::RaycastHit>::prepareForCpyLoad(&((source + i)->hit), &((hosClone + i)->hit), &((devTarget + i)->hit), 1)) break;
		hosClone[i].photon = source[i].photon;
		hosClone[i].color = source[i].color;
		hosClone[i].bounce = source[i].bounce;
		hosClone[i].bounceId = source[i].bounceId;
	}
	if (i < count) { undoCpyLoadPreparations(source, hosClone, devTarget, i); return false; }
	return true;
}
template<typename HitType>
inline void TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::undoCpyLoadPreparations(const Frame *source, Frame *hosClone, Frame *devTarget, int count) {
	for (int i = 0; i < count; i++) TypeTools<typename ShadedOctree<HitType>::RaycastHit>::undoCpyLoadPreparations(&((source + i)->hit), &((hosClone + i)->hit), &((devTarget + i)->hit), 1);
}
template<typename HitType>
inline bool TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::devArrayNeedsToBeDisposed() {
	return TypeTools<typename ShadedOctree<HitType>::RaycastHit>::devArrayNeedsToBeDisposed();
}
template<typename HitType>
inline bool TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::disposeDevArray(Frame *arr, int count) {
	for (int i = 0; i < count; i++) if (!TypeTools<typename ShadedOctree<HitType>::RaycastHit>::disposeDevArray(&((arr + i)->hit), 1)) return false;
	return true;
}


template<typename HitType, unsigned int MaxStackSize>
class TypeTools<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> > {
public:
	typedef BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> Process;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(Process);
};
template<typename HitType, unsigned int MaxStackSize>
__device__ __host__ inline void TypeTools<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> >::init(Process &m) {
	for (unsigned int i = 0; i < MaxStackSize; i++)
		TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::init(m.stack[i]);
	m.end = (m.stack + MaxStackSize);
	m.ptr = m.stack;
}
template<typename HitType, unsigned int MaxStackSize>
__device__ __host__ inline void TypeTools<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> >::dispose(Process &m) {
	for (unsigned int i = 0; i < MaxStackSize; i++)
		TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::dispose(m.stack[i]);
}
template<typename HitType, unsigned int MaxStackSize>
__device__ __host__ inline void TypeTools<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> >::swap(Process &a, Process &b) {
	TypeTools<const BackwardTracerPrivate::SceneDataHandles<HitType>*>::swap(a.world, b.world);
	for (unsigned int i = 0; i < MaxStackSize; i++)
		TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::swap(a.stack[i], b.stack[i]);
	int deltaA = (a.end - a.stack);
	int deltaB = (b.end - b.stack);
	a.end = a.stack + deltaB;
	b.end = b.stack + deltaA;
	deltaA = (a.ptr - a.stack);
	deltaB = (b.ptr - b.stack);
	a.ptr = a.stack + deltaB;
	b.ptr = b.stack + deltaA;
	TypeTools<BackwardTracerPrivate::Pixel::PixelColor>::swap(a.output, b.output);
	TypeTools<bool>::swap(a.renderComplete, b.renderComplete);
	TypeTools<bool>::swap(a.shadeStarted, b.shadeStarted);
	TypeTools<bool>::swap(a.midShade, b.midShade);
	TypeTools<Photon>::swap(a.savedPhoton, b.savedPhoton);
	TypeTools<int>::swap(a.lightId, b.lightId);
	TypeTools<int>::swap(a.castsLeft, b.castsLeft);
}
template<typename HitType, unsigned int MaxStackSize>
__device__ __host__ inline void TypeTools<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> >::transfer(Process &src, Process &dst) {
	TypeTools<const BackwardTracerPrivate::SceneDataHandles<HitType>*>::transfer(src.world, dst.world);
	for (unsigned int i = 0; i < MaxStackSize; i++)
		TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::transfer(src.stack[i], dst.stack[i]);
	int deltaA = (src.end - src.stack);
	int deltaB = (dst.end - dst.stack);
	src.end = src.stack + deltaB;
	dst.end = dst.stack + deltaA;
	deltaA = (src.ptr - src.stack);
	deltaB = (dst.ptr - dst.stack);
	src.ptr = src.stack + deltaB;
	dst.ptr = dst.stack + deltaA;
	TypeTools<BackwardTracerPrivate::Pixel::PixelColor>::transfer(src.output, dst.output);
	TypeTools<bool>::transfer(src.renderComplete, dst.renderComplete);
	TypeTools<bool>::transfer(src.shadeStarted, dst.shadeStarted);
	TypeTools<bool>::transfer(src.midShade, dst.midShade);
	TypeTools<Photon>::transfer(src.savedPhoton, dst.savedPhoton);
	TypeTools<int>::transfer(src.lightId, dst.lightId);
	TypeTools<int>::transfer(src.castsLeft, dst.castsLeft);
}
template<typename HitType, unsigned int MaxStackSize>
inline bool TypeTools<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> >::prepareForCpyLoad(const Process *source, Process *hosClone, Process *devTarget, int count) {
	int i = 0;
	for (i = 0; i < count; i++) {
		if (!TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::prepareForCpyLoad(&((source + i)->stack), &((hosClone + i)->stack), &((devTarget + i)->stack), MaxStackSize)) break;
		hosClone[i].end = ((devTarget + i)->stack + (source[i].end - source[i].stack));
		hosClone[i].ptr = ((devTarget + i)->stack + (source[i].ptr - source[i].stack));
		hosClone[i].output = source[i].output;
		hosClone[i].renderComplete = source[i].renderComplete;
		hosClone[i].shadeStarted = source[i].shadeStarted;
		hosClone[i].midShade = source[i].midShade;
		hosClone[i].savedPhoton = source[i].savedPhoton;
		hosClone[i].lightId = source[i].lightId;
		hosClone[i].castsLeft = source[i].castsLeft;
	}
	if (i < count) {
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return false;
	}
	return true;
}
template<typename HitType, unsigned int MaxStackSize>
inline void TypeTools<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> >::undoCpyLoadPreparations(const Process *source, Process *hosClone, Process *devTarget, int count) {
	for (int i = 0; i < count; i++)
		TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::undoCpyLoadPreparations(&((source + i)->stack), &((hosClone + i)->stack), &((devTarget + i)->stack), MaxStackSize);
}
template<typename HitType, unsigned int MaxStackSize>
inline bool TypeTools<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> >::devArrayNeedsToBeDisposed() {
	return TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::devArrayNeedsToBeDisposed();
}
template<typename HitType, unsigned int MaxStackSize>
inline bool TypeTools<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> >::disposeDevArray(Process *arr, int count) {
	for (int i = 0; i < count; i++)
		if (!TypeTools<BackwardTracerPrivate::PixelRenderFrame<HitType> >::disposeDevArray(&((arr + i)->stack), MaxStackSize)) return false;
	return true;
}
#undef BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_WIDTH
#undef BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_HEIGHT
#undef BACKWARD_TRACER_PRIVATE_KERNELS_BLOCK_SIZE









namespace BackwardTracerPrivate {
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
	/** ########################################################################## **/
	/** Kernels:                                                                   **/
	/** ########################################################################## **/
	template<typename HitType, unsigned int MaxStackSize>
	__dumb__ static Pixel::PixelColor renderPixel(const Photon &photon, const SceneDataHandles<HitType> &world, PixelRenderProcess<HitType, MaxStackSize> *stack) {
		PixelRenderProcess<HitType, MaxStackSize> pixel;
		pixel.setup(photon, world);
		while (!pixel.renderComplete) {
			pixel.castsLeft = 1;
			pixel.iterate();
		}
		return pixel.output;
	}

	template<typename HitType, unsigned int MaxStackSize>
	__dumb__ void colorPixel(int x, int y, int width, int height, const SceneDataHandles<HitType> &world, Pixel &pixel, PixelRenderProcess<HitType, MaxStackSize> *stack) {
		pixel.currentIteration.color = ColorRGB(0.0f, 0.0f, 0.0f);
		pixel.currentIteration.depth = 0.0f;
		float wSegs = 1.0f;
		float hSegs = 1.0f;
		float wStep = 1.0f / wSegs;
		float hStep = 1.0f / hSegs;
		float surf = wSegs * hSegs;
		for (float i = 0.0f; i < 1.0f; i += wStep)
			for (float j = 0.0f; j < 1.0f; j += hStep) {
				Vector2 screenSpacePoint = toScreenSpace((float)x + i, (float)y + j, width, height);
				PhotonPack pack;
				world.camera->getPhoton(screenSpacePoint, pack);
				Photon photon = pack[0];
				Pixel::PixelColor color = renderPixel<HitType, MaxStackSize>(photon, world, stack);
				if (color.depth >= 0) {
					pixel.currentIteration.color += color.color / surf;
					pixel.currentIteration.depth += color.depth / surf;
				}
				else {
					pixel.currentIteration.color += pixel.background / surf;
					pixel.currentIteration.depth += INFINITY / surf;
				}
			}

	}

	template<typename HitType, unsigned int MaxStackSize>
	__global__ static void renderImage(Matrix<Pixel> *imageBuffer, const SceneDataHandles<HitType> world, int blockOffset, int iteration, Stacktor<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxStackSize> > *stacks) {
		int x, y; if (!getPixelId(imageBuffer->width(), imageBuffer->height(), x, y, blockOffset)) return;
		Pixel &pixel = imageBuffer->operator[](y)[x];
		colorPixel<HitType, MaxStackSize>(x, y, imageBuffer->width(), imageBuffer->height(), world, pixel, NULL);
		float factor = (1.0f / ((float)(iteration + 1)));
		pixel.lastIteration.color += (pixel.currentIteration.color - pixel.lastIteration.color) * factor;
		pixel.lastIteration.depth += (pixel.currentIteration.depth - pixel.lastIteration.depth) * factor;
	}
}







/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** ########################################################################## **/
/** User Interface:                                                            **/
/** ########################################################################## **/
template<typename HitType, unsigned int MaxBounces>
__host__ inline BackwardTracer<HitType, MaxBounces>::BackwardTracer() {
	parameters.usingDevice = false;
	parameters.usingHost = false;
	parameters.CPUthreadLimit = -1;
	camera = NULL;
	scene = NULL;
	lights = NULL;
	iterationId = 0;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline BackwardTracer<HitType, MaxBounces>::~BackwardTracer() {
	clear();
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::clear() {
	bool success = pixels.destroyHandles();
	if (!renderProcess.destroyHandles()) success = false;
	return success;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline BackwardTracer<HitType, MaxBounces>::BackwardTracer(BackwardTracer&& b) {
	swapWith(b);
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline BackwardTracer<HitType, MaxBounces>& BackwardTracer<HitType, MaxBounces>::operator=(BackwardTracer&& b) {
	swapWith(b);
	return(*this);
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline void BackwardTracer<HitType, MaxBounces>::swapWith(BackwardTracer& b) {
	TypeTools<int>::swap(iterationId, b.iterationId);
	TypeTools<Parameters>::swap(parameters, b.parameters);
	TypeTools<Handler<Matrix<BackwardTracerPrivate::Pixel> > >::swap(pixels, b.pixels);
	TypeTools<BackwardTracerPrivate::BackwardTracerRenderProcess<HitType, MaxBounces> >::swap(renderProcess, b.renderProcess);
	TypeTools<const Handler<const Camera> *>::swap(camera, b.camera);
	TypeTools<const Handler<const ShadedOctree<HitType> > *>::swap(scene, b.scene);
	TypeTools<const Handler<const Stacktor<Light> > *>::swap(lights, b.lights);
}





template<typename HitType, unsigned int MaxBounces>
__host__ inline void BackwardTracer<HitType, MaxBounces>::setCamera(const Handler<const Camera> &cam) {
	camera = &cam;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline void BackwardTracer<HitType, MaxBounces>::setScene(const Handler<const ShadedOctree<HitType> > &geometry, const Handler<const Stacktor<Light> > &lightList) {
	scene = &geometry;
	lights = &lightList;
}





template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::useDevice(bool use) {
	if (pixels.hostHandle == NULL)
		if (!pixels.createHandle()) return false;
	if (renderProcess.hostHandle == NULL)
		if (!renderProcess.createHandle()) return false;
	if (!pixels.uploadHostHandleToDevice(true)) return false;
	if (!renderProcess.uploadHostHandleToDevice(true)) return false;
	// ETC... render process needs to altered for host...
	parameters.usingDevice = true;
	iterationId = 0;
	return true;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::useHost(bool use) {
	if (pixels.hostHandle == NULL)
		if (!pixels.createHandle()) return false;
	if (renderProcess.hostHandle == NULL)
		if (!renderProcess.createHandle()) return false;
	// ETC... render process needs to altered for host...
	parameters.usingHost = true;
	iterationId = 0;
	return true;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline void BackwardTracer<HitType, MaxBounces>::setCPUthreadLimit(int limit) {
	parameters.CPUthreadLimit = limit;
}





template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::setResolution(int width, int height) {
	if (width < 0 || height < 0) return false;
	if (pixels.hostHandle == NULL)
		if (!pixels.createHandle()) return false;
	if (width != pixels.hostHandle->width() || height != pixels.hostHandle->height())
		pixels.hostHandle->setDimensions(width, height);
	if (parameters.usingDevice)
		if (!pixels.uploadHostHandleToDevice(true)) return false;
	iterationId = 0;
	return true;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::getResolution(int &width, int &height) {
	if (pixels.hostHandle == NULL) return false;
	width = pixels.hostHandle->width();
	height = pixels.hostHandle->height();
	return true;
}





template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::cleanImage(ColorRGB background) {
	if (parameters.usingDevice) {
		if (pixels.hostHandle == NULL || pixels.deviceHandle == NULL) return false;
		cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
		int blocks, threads; BackwardTracerPrivate::getKernelDimensions(pixels.hostHandle->width(), pixels.hostHandle->height(), blocks, threads);
		BackwardTracerPrivate::cleanImage<<<blocks, threads, 0, stream>>>(pixels.deviceHandle, background);
		bool success = (cudaStreamSynchronize(stream) == cudaSuccess);
		if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
		if (success) iterationId = 0;
		return success;
	}
	if (parameters.usingHost) {
		if (pixels.hostHandle == NULL) return false;
		int width = pixels.hostHandle->width();
		int height = pixels.hostHandle->height();
		Matrix<BackwardTracerPrivate::Pixel> &matrix = (*pixels.hostHandle);
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++) {
				matrix[i][j].background = background;
				matrix[i][j].lastIteration.color = background;
				matrix[i][j].lastIteration.depth = FLT_MAX;
			}
		iterationId = 0;
	}
	return true;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::cleanImage(const Handler<Matrix<Color> >& background) {
	return false;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::cleanImage(const Handler<Matrix<ColorRGB> >& background) {
	return false;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline void BackwardTracer<HitType, MaxBounces>::resetIterations() {
	iterationId = 0;
}
#define MAX_BLOCKS_PER_KERNEL 128
template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::iterate() {
	if (parameters.usingDevice) {
		if (camera == NULL || scene == NULL || lights == NULL) return false;
		if (camera->deviceHandle == NULL || scene->deviceHandle == NULL || lights->deviceHandle == NULL) return false;
		if (pixels.hostHandle == NULL || pixels.deviceHandle == NULL) return false;
		cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
		int blocks, threads; BackwardTracerPrivate::getKernelDimensions(pixels.hostHandle->width(), pixels.hostHandle->height(), blocks, threads);
		int offset = 0;
		bool success = true;
		BackwardTracerPrivate::SceneDataHandles<HitType> handles;
		handles.camera = camera->deviceHandle;
		handles.world = scene->deviceHandle;
		handles.lights = lights->deviceHandle;
		Handler<Stacktor<BackwardTracerPrivate::PixelRenderProcess<HitType, MaxBounces> > > handler;
		//handler.createHandle();
		//handler.object().flush(MAX_BLOCKS_PER_KERNEL * threads);
		//handler.uploadHostHandleToDevice();
		while (blocks > 0) {
			int kernel_blocks = min(blocks, MAX_BLOCKS_PER_KERNEL);
			BackwardTracerPrivate::renderImage<HitType, MaxBounces><<<kernel_blocks, threads, 0, stream>>>(pixels.deviceHandle, handles, offset, iterationId, handler.deviceHandle);
			//*
			if (cudaStreamSynchronize(stream) != cudaSuccess) {
				success = false;
				break;
			}
			//*/
			offset += kernel_blocks;
			blocks -= kernel_blocks;
		}
		//handler.destroyHandles();
		if (cudaStreamSynchronize(stream) != cudaSuccess) success = false;
		if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
		if (success) iterationId++;
		return success;
	}
	if (parameters.usingHost) {
		if (camera == NULL || scene == NULL || lights == NULL) return false;
		if (camera->hostHandle == NULL || scene->hostHandle == NULL || lights->hostHandle == NULL) return false;
		if (pixels.hostHandle == NULL) return false;
		//int width = pixels.hostHandle->width();
		//int height = pixels.hostHandle->height();
		//Matrix<BackwardTracerPrivate::Pixel> &matrix = (*pixels.hostHandle);
		//BackwardTracerPrivate::SceneDataHandles<HitType> handles;
		//handles.camera = camera->hostHandle;
		//handles.world = scene->hostHandle;
		//handles.lights = lights->hostHandle;
		//iterationId++;
		return true;
	}
	return false;
}
#undef MAX_BLOCKS_PER_KERNEL
template<typename HitType, unsigned int MaxBounces>
__host__ inline int BackwardTracer<HitType, MaxBounces>::iteration()const {
	return iterationId;
}





template<typename HitType, unsigned int MaxBounces>
template<typename ColorType>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::loadOutput(Handler<Matrix<ColorType> >& destination)const {
	if (parameters.usingDevice) {
		if (pixels.hostHandle == NULL || pixels.deviceHandle == NULL || destination.deviceHandle == NULL) return false;
		cudaStream_t stream; if (cudaStreamCreate(&stream) != cudaSuccess) return false;
		int blocks, threads; BackwardTracerPrivate::getKernelDimensions(pixels.hostHandle->width(), pixels.hostHandle->height(), blocks, threads);
		BackwardTracerPrivate::loadImage<ColorType><<<blocks, threads, 0, stream>>>(pixels.deviceHandle, destination.deviceHandle);
		bool success = (cudaStreamSynchronize(stream) == cudaSuccess);
		if (cudaStreamDestroy(stream) != cudaSuccess) success = false;
		return success;
	}
	if (parameters.usingHost) {
		if (pixels.hostHandle == NULL || destination.hostHandle == NULL) return false;
		const Matrix<BackwardTracerPrivate::Pixel> &origin = (*pixels.hostHandle);
		Matrix<ColorType> &dest = (*destination.hostHandle);
		int width = min(origin.width(), dest.width());
		int height = min(origin.height(), dest.height());
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				dest[i][j] = origin[i][j].lastIteration.color;
	}
	return true;
}

