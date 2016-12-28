#pragma once
#include"cuda_runtime.h"
#include"device_launch_parameters.h"
#include<iostream>



namespace Device {
	static void dumpCurrentDevice();
}




#include"Device.impl.cuh"
