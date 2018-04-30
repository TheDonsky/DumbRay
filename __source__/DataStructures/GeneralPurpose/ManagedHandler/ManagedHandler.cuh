#pragma once
#include "../Stacktor/Stacktor.cuh"
#include <mutex>
#include <thread>



template <typename Type>
class ManagedHandler {
public:
	/*
	Constructor.
	*/
	inline ManagedHandler(const Type& s);

	/*
	Destructor.
	*/
	inline ~ManagedHandler();

	/*
	Sets the GPU context.
	*/
	inline bool selectGPU(int index)const;

	/*
	Uploads/updates the item to the given GPU and selects it's context.
	*/
	inline bool uploadToGPU(int index, bool overrideExisting = false);

	/*
	Deallocates the GPU instance (leaves context set).
	*/
	inline bool cleanGPU(int index);

	/*
	Uploads/updates the item on every available GPU (which context will stay selected, is undefined).
	*/
	inline void uploadToEveryGPU(bool overrideExisting = false);

	/*
	Deallocates every GPU instance (which context will stay selected, is undefined).
	*/
	inline void cleanEveryGPU();

	/*
	Returns detected GPU count.
	*/
	inline int gpuCount()const;

	/*
	Returns CPU handle.
	*/
	inline const Type* getHandleCPU()const;

	/*
	Returns GPU handle (no context selection here...).
	*/
	inline Type* getHandleGPU(int index);

	/*
	Returns GPU handle (no context selection here...).
	*/
	inline const Type* getHandleGPU(int index)const;





private:
	//std::mutex lock;
	const Type*data;
	Stacktor<Type*> deviceData;

	// WE RESTRICT COPY-CONSTRUCTION FOR THIS ONE...
	inline ManagedHandler(const ManagedHandler &other) {}
	inline ManagedHandler &operator=(const ManagedHandler &other) { return (*this); }

	inline static void cleanDeviceInstanceThread(ManagedHandler *self, int deviceId);
	inline static void createDeviceInstandeThread(ManagedHandler *self, int deviceId, bool overrideExisting);
};





#include "ManagedHandler.impl.cuh"
