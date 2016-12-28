#include"Device.cuh"




namespace Device {
	static void dumpCurrentDevice() {
		int device;
		if (cudaGetDevice(&device) != cudaSuccess) {
			std::cout << "ERROR: Unable to detect CUDA device" << std::endl;
			return;
		}
		cudaDeviceProp properties;
		if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
			std::cout << "ERROR: Unable to retrieve CUDA device properties (id = " << device << ")" << std::endl;
			return;
		}
		std::cout << "####################################################################" << std::endl;
		std::cout << "------------ Device:                      " << device << std::endl;
		std::cout << "------------ Name:                        " << properties.name << std::endl;
		std::cout << "------------ Total Global Memory:         " << properties.totalGlobalMem << std::endl;
		std::cout << "------------ Shared Memory per block:     " << properties.sharedMemPerBlock << std::endl;
		std::cout << "------------ Registers per block:         " << properties.regsPerBlock << std::endl;
		std::cout << "------------ Warp Size:                   " << properties.warpSize << std::endl;
		std::cout << "------------ Memory Pitch:                " << properties.memPitch << std::endl;
		std::cout << "------------ Max Threads Per Block:       " << properties.maxThreadsPerBlock << std::endl;
		std::cout << "------------ Max Threads dim:             [" << properties.maxThreadsDim[0] << ", " << properties.maxThreadsDim[1] << ", " << properties.maxThreadsDim[2] << "]" << std::endl;
		std::cout << "------------ Max Grid Size:               [" << properties.maxGridSize[0] << ", " << properties.maxGridSize[1] << ", " << properties.maxGridSize[2] << "]" << std::endl;
		std::cout << "------------ Total Constant Memory:       " << properties.totalConstMem << std::endl;
		std::cout << "------------ Compute Capability:          " << properties.major << "." << properties.minor << std::endl;
		std::cout << "------------ Clock Rate:                  " << properties.clockRate << std::endl;
		std::cout << "------------ Texture Alignment:           " << properties.textureAlignment << std::endl;
		std::cout << "------------ Device Overlap:              " << properties.deviceOverlap << std::endl;
		std::cout << "------------ Multi Processor Count:       " << properties.multiProcessorCount << std::endl;
		std::cout << "------------ Kernel Exec Timeout Enabled: " << properties.kernelExecTimeoutEnabled << std::endl;
		std::cout << "------------ Integrated:                  " << properties.integrated << std::endl;
		std::cout << "------------ Can Map Host Memory:         " << properties.canMapHostMemory << std::endl;
		std::cout << "------------ Compute Mode:                " << properties.computeMode << std::endl;
		std::cout << "------------ Concurrent Kernels:          " << properties.concurrentKernels << std::endl;
		std::cout << "------------ ECC Enabled:                 " << properties.ECCEnabled << std::endl;
		std::cout << "------------ PCI Bus ID:                  " << properties.pciBusID << std::endl;
		std::cout << "------------ PCI Device ID:               " << properties.pciDeviceID << std::endl;
		std::cout << "------------ TCC Driver:                  " << properties.tccDriver << std::endl;
		std::cout << "####################################################################" << std::endl;
	}
}
