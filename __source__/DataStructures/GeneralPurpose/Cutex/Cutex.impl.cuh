#include"Cutex.cuh"


__device__ __host__ inline Cutex::Cutex(){
	initRaw();
}
__device__ __host__ inline void Cutex::initRaw(){
	state = 1;
#ifndef __CUDA_ARCH__
	std::mutex *mut = ((std::mutex*)mutexBytes);
	new(mut)std::mutex();
#endif
}
__device__ __host__ inline Cutex::~Cutex(){
	disposeRaw();
}
__device__ __host__ inline void Cutex::disposeRaw(){
	state = 1;
#ifndef __CUDA_ARCH__
	std::mutex *mut = ((std::mutex*)mutexBytes);
	mut->~mutex();
#endif
}

__device__ __host__ inline void Cutex::init(int startState){
	if (startState < 0) startState = 0;
	if (startState > 1) startState = 1;
	state = startState;
}
__device__ __host__ inline void Cutex::lock(){
#ifndef __CUDA_ARCH__
	std::mutex *mut = ((std::mutex*)mutexBytes);
	mut->lock();
#else
	while(atomicCAS((int*)&state, 1, 0) != 1);
#endif
}
__device__ __host__ inline void Cutex::unlock(){
#ifndef __CUDA_ARCH__
	std::mutex *mut = ((std::mutex*)mutexBytes);
	mut->unlock();
#else
	atomicExch((int*)&state, 1);
#endif
}
#define MAX_LOCK_TRY_COUNT (1<<30)
template<typename ReturnType, typename Function, typename... Args>
__device__ __host__ inline ReturnType Cutex::atomicCall(Function&& func, Args&... args){
	ReturnType rv;
#ifndef __CUDA_ARCH__
	std::mutex *mut = ((std::mutex*)mutexBytes);
	mut->lock();
	rv = (func(args...));
	mut->unlock();
#else
	int i = 0;
	while (i <= MAX_LOCK_TRY_COUNT){
		if (atomicCAS((int*)&state, 1, 0) == 1){
			rv = (func(args...));
			i = MAX_LOCK_TRY_COUNT;
			atomicExch((int*)&state, 1);
		}
		i++;
	}
#endif
	return(rv);
}
#undef MAX_LOCK_TRY_COUNT