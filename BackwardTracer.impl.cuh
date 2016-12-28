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
	__dumb__ int blockWidth() { return 16; }
	__dumb__ int blockHeight() { return 8; }
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

	__dumb__ Vector2 toScreenSpace(int x, int y, int width, int height) {
		return ((Vector2((float)x, (float)(height - y)) - (Vector2((float)width, (float)height) * 0.5f)) / ((float)height));
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

	template<typename HitType>
	struct PixelRenderFrame {
		Photon photon;
		typename ShadedOctree<HitType>::RaycastHit hit;
		ColorRGB color;
		ShaderBounce bounce;
		int bounceId;
	};

	template<typename HitType, unsigned int MaxStackSize>
	struct PixelRenderProcess {
		SceneDataHandles<HitType> world;
		PixelRenderFrame<HitType> stack[MaxStackSize];
		PixelRenderFrame<HitType> *end;
		PixelRenderFrame<HitType> *ptr;
		Pixel::PixelColor output;
		bool renderComplete;

		bool shadeStarted;
		bool midShade;
		Photon savedPhoton;
		int lightId;

		__dumb__ bool shade(int &maxRaycasts) {
			PixelRenderFrame<HitType> &frame = (*ptr);
			
			if (!shadeStarted) {
				if (maxRaycasts <= 0) {
					midShade = true;
					return true;
				}
				else midShade = false;
				if (frame.photon.dead()) return false;
				if (!world.world->cast(frame.photon.ray, frame.hit, false)) return false;
				shadeStarted = true;
				maxRaycasts--;

				frame.color(0.0f, 0.0f, 0.0f);

				// BOUNCE FROM INDIRECT ILLUMINATION:
				if ((ptr + 1) != end) {
					ShaderBounceInfo<HitType> bounceInfo = { frame.hit.object.object, frame.photon, frame.hit.hitPoint };
					frame.hit.object.material->bounce(bounceInfo, &frame.bounce);
				}
				else frame.bounce.count = 0;
				frame.bounceId = 0;
			}

			// COLOR FROM LIGHT SOURCES:
			const Stacktor<Light> &lights = (*world.lights);
			for (int i = lightId; i < lights.size(); i++) {
				bool noShadows;
				Photon p;
				if (midShade) {
					noShadows = false;
					p = savedPhoton;
				}
				else {
					noShadows = false;
					p = lights[i].getPhoton(frame.hit.hitPoint, &noShadows);
					if (p.dead()) continue;
				}
				if (!noShadows) {
					if (maxRaycasts > 0) {
						typename ShadedOctree<HitType>::RaycastHit lightHit;
						if (world.world->cast(p.ray, lightHit, false)) {
							if ((frame.hit.hitPoint - lightHit.hitPoint).sqrMagnitude() <= 128.0f * VECTOR_EPSILON)
								noShadows = true;
						}
						else noShadows = true;
						midShade = false;
						maxRaycasts--;
					}
					else {
						midShade = true;
						savedPhoton = p;
						lightId = i;
						return true;
					}
				}
				if (noShadows) {
					ShaderHitInfo<HitType> castInfo = { frame.hit.object.object, p, frame.hit.hitPoint, frame.photon.ray.origin };
					frame.color += frame.hit.object.material->illuminate(castInfo).color;
				}
			}
			frame.color *= frame.photon.color;

			shadeStarted = false;
			midShade = false;
			lightId = 0;
			return true;
		}

		__dumb__ bool iterate(int maxRaycasts) {
			while (true) {
				if (midShade || (ptr->bounceId < ptr->bounce.count)) {
					if (!midShade) {
						Photon sample = ptr->bounce.samples[ptr->bounceId];
						sample.ray.origin += sample.ray.direction * (128.0f * VECTOR_EPSILON);
						ptr->bounceId++;
						ptr++;
						ptr->photon = sample;
					}
					if (!shade(maxRaycasts)) ptr--;
					else if (midShade) return false;
				}
				else if(ptr == stack) {
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

		__dumb__ bool setup(const Photon &photon, const SceneDataHandles<HitType> &world, int maxRaycasts) {
			this->world = world;
			end = (stack + MaxStackSize);
			ptr = stack;
			ptr->photon = photon;
			shadeStarted = false;
			midShade = false;
			lightId = 0;
			if (!shade(maxRaycasts)) {
				output.depth = -1;
				renderComplete = true;
				return true;
			}
			else {
				renderComplete = false;
				iterate(maxRaycasts);
				return renderComplete;
			}
		}
	};

	template<typename HitType, unsigned int MaxStackSize>
	__dumb__ static Pixel::PixelColor renderPixel(const Photon &photon, const SceneDataHandles<HitType> &world) {
		PixelRenderProcess<HitType, MaxStackSize> process;
		process.setup(photon, world, 1);
		while (!process.renderComplete)
			process.iterate(1);
		return process.output;
	}

	template<typename HitType, unsigned int MaxStackSize>
	__dumb__ void colorPixel(int x, int y, int width, int height, const SceneDataHandles<HitType> &world, Pixel &pixel) {
		Vector2 screenSpacePoint = toScreenSpace(x, y, width, height);
		Photon photon = world.camera->getPhoton(screenSpacePoint);
		Pixel::PixelColor color = renderPixel<HitType, MaxStackSize>(photon, world);
		if (color.depth >= 0) pixel.currentIteration = color;
		else pixel.currentIteration = Pixel::PixelColor{ pixel.background, INFINITY };
	}

	template<typename HitType, unsigned int MaxStackSize>
	__global__ static void renderImage(Matrix<Pixel> *imageBuffer, const SceneDataHandles<HitType> world, int blockOffset, int iteration) {
		int x, y; if (!getPixelId(imageBuffer->width(), imageBuffer->height(), x, y, blockOffset)) return;
		Pixel &pixel = imageBuffer->operator[](y)[x];
		colorPixel<HitType, MaxStackSize>(x, y, imageBuffer->width(), imageBuffer->height(), world, pixel);
		float factor = (1.0f / ((float)(iteration + 1)));
		pixel.lastIteration.color += (pixel.currentIteration.color - pixel.lastIteration.color) * factor;
		pixel.lastIteration.depth += (pixel.currentIteration.depth - pixel.lastIteration.depth) * factor;
	}
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
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
	return pixels.destroyHandles();
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
	if (!pixels.uploadHostHandleToDevice(true)) return false;
	parameters.usingDevice = true;
	iterationId = 0;
	return true;
}
template<typename HitType, unsigned int MaxBounces>
__host__ inline bool BackwardTracer<HitType, MaxBounces>::useHost(bool use) {
	if (pixels.hostHandle == NULL)
		if (!pixels.createHandle()) return false;
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
#define MAX_BLOCKS_PER_KERNEL 512
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
		while (blocks > 0) {
			int kernel_blocks = min(blocks, MAX_BLOCKS_PER_KERNEL);
			BackwardTracerPrivate::renderImage<HitType, MaxBounces><<<kernel_blocks, threads, 0, stream>>>(pixels.deviceHandle, handles, offset, iterationId);
			//*
			if (cudaStreamSynchronize(stream) != cudaSuccess) {
				success = false;
				break;
			}
			//*/
			offset += kernel_blocks;
			blocks -= kernel_blocks;
		}
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

