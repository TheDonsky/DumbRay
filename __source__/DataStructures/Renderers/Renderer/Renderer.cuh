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
			ONE = 1		// Single thread.
		};

	public:
		/*
		Constructor.
		(By default, allocates all CPU and GPU resources, using a single thread per GPU).
		*/
		ThreadConfiguration();

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

	private:
		friend class Renderer;
		int threadsOnCPU;
		Stacktor<int> threadsPerGPU;
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
		void *sharedData;
		void *data;
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
		ITERATE,
		QUIT
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

	volatile int iterationId;

	volatile bool threadsStarted;
	volatile bool destructorCalled;
	volatile Command command;
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
	static void thread(Renderer *renderer, ThreadAttributes *attributes);
};
