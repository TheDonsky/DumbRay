#pragma once
#include<time.h>
#include"AABB.h"


namespace AABB_Test{

#define ITERATIONS 4000
	__global__ void testOnCuda(AABB box, Vector3 a, Vector3 b, Vector3 c, bool *res){
		for (int i = 0; i < ITERATIONS; i++) (*res) = box.intersectsTriangle(a, b, c);
		if (threadIdx.x == 0 && blockIdx.x == 0) printf("%d\n", (int)(box.intersectsTriangle(a, b, c)));
	}
	
	void triangles(){
		AABB box(Vector3::zero(), Vector3::one());
		while (true){
			std::cout << "Enter the triangle: " << std::endl;
			std::cout << "A: "; Vector3 a; std::cin >> a;
			std::cout << "B: "; Vector3 b; std::cin >> b;
			std::cout << "C: "; Vector3 c; std::cin >> c;
			if (box.intersectsTriangle(a, b, c))
				std::cout << "Intersects." << std::endl;
			else std::cout << "Does not intersect." << std::endl;

			const int n = 100000000;
			std::cout << "doing the same " << n << " times..." << std::endl;
			long long t = clock();
			for (int i = 0; i < n; i++)
				box.intersectsTriangle(a, b, c);
			std::cout << "Done. (" << (clock() - t) << "ms)" << std::endl;
			const unsigned int numBlocks = 10000;
			const unsigned int numThreads = 512;
			std::cout << "Doing same kind of shit on the device(" << ((long long)numBlocks * (long long)numThreads * (long long)ITERATIONS) << " times(combined))..." << std::endl;
			t = clock();
			bool *res;
			cudaMalloc(&res, sizeof(bool));
			testOnCuda<<<numBlocks, numThreads>>>(box, a, b, c, res);
			cudaDeviceSynchronize();
			bool r; if (cudaMemcpy(&r, res, sizeof(bool), cudaMemcpyDeviceToHost) != cudaSuccess) std::cout << "ERROR" << std::endl;
			cudaFree(res);
			std::cout << "Result: " << r << std::endl;
			std::cout << "Done. (" << (clock() - t) << "ms)" << std::endl;


			bool ans;
			while (true){
				std::cout << "Do you want to quit this test? (Y/N): ";
				char c; std::cin >> c;
				if (c == 'Y' || c == 'y'){ ans = true; break; }
				else if (c == 'N' || c == 'n'){ ans = false; break; }
			}
			if (ans) break;
		}
	}
#undef ITERATIONS
}
