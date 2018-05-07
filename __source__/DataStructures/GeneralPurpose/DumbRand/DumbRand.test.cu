#include"DumbRand.test.cuh"
#include"DumbRand.cuh"
#include<iostream>
#include<string>


namespace DumbRandTest {
	namespace {
		template<typename FunctionType, typename... Args>
		__device__ __host__ inline static void generateAndPrint(const char *comment, const char *typeHint, DumbRand &generator, FunctionType&& generate, Args... args) {
			printf("%s\n", comment);
			for (int i = 0; i < 16; i++) {
				printf(typeHint, generate(generator, args...));
				printf(" ");
				printf(typeHint, generate(generator, args...));
				printf(" ");
				printf(typeHint, generate(generator, args...));
				printf(" ");
				printf(typeHint, generate(generator, args...));
				printf(" ");
				printf(typeHint, generate(generator, args...));
				printf(" ");
				printf(typeHint, generate(generator, args...));
				printf(" ");
				printf(typeHint, generate(generator, args...));
				printf(" ");
				printf(typeHint, generate(generator, args...));
				printf("\n");
			}
			//	std::cout << generate(generator, args...) << " " << generate(generator, args...) << " " << generate(generator, args...) << " " << generate(generator, args...) << std::endl;
			//std::cout << std::endl;
			printf("\n");
		}

		template<typename FunctionType, typename... Args>
		void generateAndPrintStatistics(const std::string &comment, int start, int end, int sampleCount, FunctionType&& generate, Args... args) {
			std::cout << comment << std::endl;
			int possibleOutputs = (end - start + 1);
			int *counts = new int[possibleOutputs];
			for (int i = 0; i < possibleOutputs; i++) counts[i] = 0;
			DumbRand generator;
			generator.seed();
			std::cout << "Generating " << sampleCount << " samples between " << start << " and " << end << "...." << std::endl;
			for (int i = 0; i < sampleCount; i++)
				counts[((int)generate(generator, start, end, args...)) - start]++;
			for (int i = 0; i < possibleOutputs; i++)
				std::cout << (i + start) << ": " << counts[i] << std::endl;
			std::cout << std::endl;
			delete[] counts;
		}

		__device__ __host__ inline static DumbRand::UnsignedInt getUnsigned(DumbRand &generator) { return generator.get(); }
		__device__ __host__ inline static DumbRand::SignedInt getSigned(DumbRand &generator) { return generator.getInt(); }
		__device__ __host__ inline static float getFloat(DumbRand &generator) { return generator.getFloat(); }
		
		__device__ __host__ inline static DumbRand::UnsignedInt getUnsignedRange(DumbRand &generator, uint32_t start, uint32_t end) { return generator.rangeUnsigned(start, end); }
		__device__ __host__ inline static DumbRand::SignedInt getSignedRange(DumbRand &generator, int start, int end) { return generator.rangeSigned(start, end); }
		__device__ __host__ inline static float getFloatRange(DumbRand &generator, float start, int end) { return generator.range(start, end); }

		__global__ static void getOnKernel(DumbRand generator) {
			generateAndPrint("Generating some unsigned numbers with DumbRand (Kernel):", "%u", generator, getUnsigned);
			generateAndPrint("Generating some signed numbers with DumbRand (Kernel):", "%d", generator, getSigned);
			generateAndPrint("Generating some floating point numbers with DumbRand (Kernel):", "%f", generator, getFloat);
			generateAndPrint("Generating some unsigned numbers from 5 to 10 with DumbRand (Kernel):", "%u", generator, getUnsignedRange, 5, 10);
			generateAndPrint("Generating some signed numbers from 5 to 10 with DumbRand (Kernel):", "%d", generator, getSignedRange, 5, 10);
			generateAndPrint("Generating some floating point numbers from 5 to 10 with DumbRand (Kernel):", "%f", generator, getFloatRange, 5, 10);
		}
	}

	void generateUnsigned() {
		DumbRand generator; generator.seed();
		generateAndPrint("Generating some unsigned numbers with DumbRand:", "%u", generator, getUnsigned);
	}
	void generateSigned() {
		DumbRand generator; generator.seed();
		generateAndPrint("Generating some signed numbers with DumbRand:", "%d", generator, getSigned);
	}
	void generateFloat() {
		DumbRand generator; generator.seed();
		generateAndPrint("Generating some floating point numbers with DumbRand:", "%f", generator, getFloat);
	}

	void generateUnsignedRange() {
		DumbRand generator; generator.seed(); 
		generateAndPrint("Generating some unsigned numbers from 5 to 10 with DumbRand:", "%u", generator, getUnsignedRange, 5, 10);
	}
	void generateSignedRange() {
		DumbRand generator; generator.seed(); 
		generateAndPrint("Generating some signed numbers from 5 to 10 with DumbRand:", "%d", generator, getSignedRange, 5, 10);
	}
	void generateFloatRange() {
		DumbRand generator; generator.seed(); 
		generateAndPrint("Generating some floating point numbers from 5 to 10 with DumbRand:", "%f", generator, getFloatRange, 5, 10);
	}

	void countStatisticsUnsigned() { generateAndPrintStatistics("Counting statistics for unsigned numbers", 4, 32, 802402400, getUnsignedRange); }
	void countStatisticsSigned() { generateAndPrintStatistics("Counting statistics for signed numbers", 4, 32, 802402400, getSignedRange); }
	void countStatisticsFloat() { generateAndPrintStatistics("Counting statistics for floating point numbers", 4, 32, 802402400, getFloatRange); }
	
	void generateOnKernel() {
		DumbRand generator; generator.seed();
		getOnKernel<<<1, 1>>>(generator);
		cudaDeviceSynchronize();
	}

	void test() {
		generateUnsigned();
		generateSigned();
		generateFloat();

		generateUnsignedRange();
		generateSignedRange();
		generateFloatRange();

		countStatisticsUnsigned();
		countStatisticsSigned();
		countStatisticsFloat();
		
		generateOnKernel();
	}
}


