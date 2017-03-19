#pragma once
#include"ShadedOctree.cuh"
#include"Camera.cuh"
#include"Light.cuh"
#include"Matrix.h"
#include"Handler.cuh"
#include"Cutex.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
namespace BackwardTracerPrivate {
	struct Pixel;
	template<typename HitType, unsigned int MaxStackSize>
	struct BackwardTracerRenderProcess;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType, unsigned int MaxBounces = 4>
class BackwardTracer {
public:
	__host__ inline BackwardTracer();
	__host__ inline ~BackwardTracer();
	__host__ inline bool clear();
	__host__ inline BackwardTracer(BackwardTracer&& b);
	__host__ inline BackwardTracer& operator=(BackwardTracer&& b);
	__host__ inline void swapWith(BackwardTracer& b);

	__host__ inline void setCamera(const Handler<const Camera> &cam);
	__host__ inline void setScene(const Handler<const ShadedOctree<HitType> > &geometry, const Handler<const Stacktor<Light> > &lightList);

	__host__ inline bool useDevice(bool use = true);
	__host__ inline bool useHost(bool use = true);
	__host__ inline void setCPUthreadLimit(int limit);

	__host__ inline bool setResolution(int width, int height);
	__host__ inline bool getResolution(int &width, int &height);

	__host__ inline bool cleanImage(ColorRGB background);
	__host__ inline bool cleanImage(const Handler<Matrix<Color> >& background);
	__host__ inline bool cleanImage(const Handler<Matrix<ColorRGB> >& background);
	__host__ inline void resetIterations();
	__host__ inline bool iterate();
	__host__ inline int iteration()const;

	template<typename ColorType>
	__host__ inline bool loadOutput(Handler<Matrix<ColorType> >& destination)const;





private:
	struct Parameters {
		bool usingDevice;
		bool usingHost;
		int CPUthreadLimit;
	};
	int iterationId;
	Parameters parameters;
	Handler<Matrix<BackwardTracerPrivate::Pixel> > pixels;
	Handler<BackwardTracerPrivate::BackwardTracerRenderProcess<HitType, MaxBounces> > renderProcess;

	const Handler<const Camera> *camera;
	const Handler<const ShadedOctree<HitType> > *scene;
	const Handler<const Stacktor<Light> > *lights;

	__host__ inline BackwardTracer(const BackwardTracer&) = delete;
	__host__ inline BackwardTracer& operator=(const BackwardTracer&) = delete;
};


typedef BackwardTracer<BakedTriFace> DumbBackTracer;







#include"BackwardTracer.impl.cuh"

