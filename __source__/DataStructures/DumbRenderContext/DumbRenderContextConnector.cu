#include "DumbRenderContextConnector.cuh"
#include "DumbRenderContext.cuh"




extern "C" void *makeContext() {
	//return NULL;
	return new DumbRenderContext();
}




