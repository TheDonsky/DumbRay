#pragma once
#include "../../GeneralPurpose/Stacktor/Stacktor.cuh"
#include "../../GeneralPurpose/Semaphore/Semaphore.h"
#include <thread>


class Renderer {
public:
	class ThreadConfiguration{
	public:
		/*
		Default configurations for the thread count;
		You are free to use any other number for more exact count.
		*/
		enum Configuration {
			ALL = -1,	// Same as ONE for GPU; std::thread::hardware_concurrency() for CPU.
			NONE = 0,	// No threads (deallocates CPU/GPU).
			ONE = 1,	// Single thread.
			ALL_BUT_THREAD_PER_GPU = -2,	// Applies to cpu only; removes a thread per active GPU.
			ALL_BUT_GPU_THREADS = -3,		// Applies to cpu only; removes a thread per each GPU thread.
		};

	public:
		/*
		Constructor.
		(By default, allocates all CPU and GPU resources, using a single thread per GPU).
		*/
		ThreadConfiguration();
		ThreadConfiguration(int cpuThreads, int threadsPerDevice);
		static ThreadConfiguration cpuOnly(int threads = ALL);
		static ThreadConfiguration gpuOnly(int threads = ONE);

		/*
		Sets the thread count for CPU.
		(You are free to use Configuration enumeration, if you wish to).
		*/
		void configureCPU(int threads = ALL);

		/*
		Sets the thread count for the given GPU.
		(You are free to use Configuration enumeration, if you wish to).
		*/
		void configureGPU(int GPU, int threads = ONE);

		/*
		Sets the thread count for every GPU.
		(You are free to use Configuration enumeration, if you wish to).
		*/
		void configureEveryGPU(int threads = ONE);

		/*
		Number of active devices:
		*/
		int numActiveDevices()const;

		/*
		Number of devices:
		*/
		int numDevices()const;

		/*
		Number of threads on the given device:
		*/
		int numDeviceThreads(int deviceId)const;

		/*
		Total number of device threads:
		*/
		int numDeviceThreads()const;

		/*
		Number of host threads:
		*/
		int numHostThreads()const;





	private:
		friend class Renderer;
		volatile int threadsOnCPU;
		Stacktor<int> threadsPerGPU;
		volatile uint8_t flags;
		void fixCpuThreadCount();
	};


public:
	/*
	Constructor;
	Takes basic thread configuration.
	*/
	Renderer(const ThreadConfiguration &config);

	/*
	Destructor;
	SHOULD BE OBERLOADED and SHOULD CALL killRenderThreads for everything to function properly.
	*/
	virtual ~Renderer();

	/*
	Returns true, if no internal error occured inside the constructor.
	(Does not guarantee, that absolutely everything is ok; usually, this won't even be worth checking.)
	*/
	bool configured()const;

	/*
	Returns the iteration counter.
	*/
	int iteration()const;
	
	/*
	Resets iteration counter.
	*/
	void resetIterations();
	
	/*
	Makes a single iteration.
	*/
	bool iterate();

	/*
	Number of active device threads:
	*/
	int deviceThreadCount();

	/*
	Number of total host threads:
	*/
	int hostThreadCount();

	/*
	Thread configuration:
	*/
	ThreadConfiguration &threadConfiguration();

	/*
	Thread configuration:
	*/
	const ThreadConfiguration &threadConfiguration()const;

	/*
	Call this to interrupt render process and "skip" iteration:
	*/
	void interruptRender();

	/*
	Call this to cancel whatever interruptRender did:
	*/
	void uninterruptRender();

	/*
	You may check, if killRenderThreads() was already called by calling this:
	*/
	bool renderInterrupted()const;





protected:
	
	/*
	This is effectively THE destructor and it should be called from the concrete object's destructor 
	in order to avoid memory leaks and/or random crashes.
	*/
	void killRenderThreads();





protected:
	struct Info {
		int device;
		int numDeviceThreads;
		int deviceThreadId;
		int globalThreadId;
		bool manageSharedData;
		bool isGPU()const;
		void * sharedData;
		void * data;

		template<typename Type>
		inline static Type* convert(void *address) { return ((Type*)address); }
		template<typename Type>
		inline Type *getSharedData()const { return convert<Type>(sharedData); }
		template<typename Type>
		inline Type *getData()const { return convert<Type>(data); }
	};

	virtual bool setupSharedData(const Info &info, void *& sharedData) = 0;
	virtual bool setupData(const Info &info, void *& data) = 0;
	virtual bool prepareIteration() = 0;
	virtual void iterateCPU(const Info &info) = 0;
	virtual void iterateGPU(const Info &info) = 0;
	virtual bool completeIteration() = 0;
	virtual bool clearData(const Info &info, void *& data) = 0;
	virtual bool clearSharedData(const Info &info, void *& sharedData) = 0;


private:
	enum Command {
		ITERATE = 0,
		QUIT = 1,
		INTERRUPT_RENDER = 2
	};
	struct Device {
		Semaphore setup;
		Semaphore clear;
		int firstThread;
		int threadCount;
		volatile bool setupFailed;
		volatile bool cleanFailed;
	};
	struct ThreadAttributes {
		Info info;
		Semaphore startLock;
		Semaphore endLock;
		Device *device;
		volatile bool setupFailed;
		volatile bool cleanFailed;
	};
	struct Thread {
		std::thread thread;
		ThreadAttributes properties;
	};
	
	struct ThreadParams {
		Renderer *renderer;
		ThreadAttributes *attributes;
	};

	ThreadConfiguration configuration;

	volatile int iterationId;

	volatile bool threadsStarted;
	volatile bool destructorCalled;
	volatile uint8_t command;
	int cpuThreads;
	int gpuThreads;
	int totalThreadCount;
	Device cpuLocks;
	Device *gpuLocks;
	Thread *threads;

	Renderer(const Renderer &other);
	Renderer& operator=(const Renderer &other);

	void startRenderThreads();
	void threadCPU(ThreadAttributes *attributes);
	void threadGPU(ThreadAttributes *attributes);
	static void thread(ThreadParams params);
};
