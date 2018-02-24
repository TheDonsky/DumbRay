#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<iostream>



namespace Device {
	void dumpCurrentDevice();
	int multiprocessorCount();
}



