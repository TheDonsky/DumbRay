#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<iostream>
#include<string>



namespace Device {
	void dumpCurrentDevice();
	int multiprocessorCount();
	std::string getDeviceName(int device);
}



