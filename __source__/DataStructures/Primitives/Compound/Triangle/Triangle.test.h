#pragma once
#include"Triangle.h"

namespace TriangleTest{
#define ITERATIONS 200000
	__global__ void testOnCuda(Triangle tri, Vertex x, int n, Vertex *res){
		for (int i = 0; i < n; i++)
			(*res) = tri.getMasses(x);
		if (threadIdx.x == 0 && blockIdx.x == 0) printf("(%f, %f, %f)\n", res->x, res->y, res->z);
	}

#define MAX_COORD 99999
	void test(){
		std::cout << "Triangle test started..." << std::endl;
		long long t = clock();
		float maxDelta = 0;
		for (int i = 0; i < 1000000; i++){
			Vertex a((float)(rand() % MAX_COORD), (float)(rand() % MAX_COORD), (float)(rand() % MAX_COORD));
			Vertex b((float)(rand() % MAX_COORD), (float)(rand() % MAX_COORD), (float)(rand() % MAX_COORD));
			Vertex c((float)(rand() % MAX_COORD), (float)(rand() % MAX_COORD), (float)(rand() % MAX_COORD));
			Triangle tri(a, b, c);
			Vertex x = tri.projectVertex(Vector3((float)(rand() % MAX_COORD), (float)(rand() % MAX_COORD), (float)(rand() % MAX_COORD)));
			Vertex m = tri.getMasses(x);
			Vertex y = tri.massCenter(m);
			float error = (y - x).magnitude();
			if (maxDelta < error) maxDelta = error;
			if (error > 2.5f){
				std::cout << "##############################################" << std::endl;
				std::cout << "Iteration No: " << i << std::endl;
				std::cout << "A: " << tri.a << std::endl;
				std::cout << "B: " << tri.b << std::endl;
				std::cout << "C: " << tri.c << std::endl << std::endl;
				std::cout << "M: " << m << std::endl << std::endl;
				std::cout << "X: " << x << std::endl;
				std::cout << "Y: " << y << std::endl;
				std::cout << "Delta: " << (x - y) << std::endl;
				std::cout << "Error: " << error << std::endl << std::endl;
				system("PAUSE");
			}
		}
		std::cout << "Max error: " << maxDelta << std::endl;
		std::cout << "correctness test finished. Clock: " << (clock() - t) << std::endl << std::endl;
		while (true){
			Triangle tri;
			std::cout << "Enter A: "; std::cin >> tri.a;
			std::cout << "Enter B: "; std::cin >> tri.b;
			std::cout << "Enter C: "; std::cin >> tri.c;
			Vertex x;
			std::cout << "Enter X: "; std::cin >> x;
			Vertex m = tri.getMasses(x);
			std::cout << "Masses: " << m << std::endl;
			std::cout << "X:            " << x << std::endl;
			std::cout << "Mass Center:  " << tri.massCenter(m) << std::endl;

			const int n = 100000000;
			std::cout << "Doing the same " << n << " times" << std::endl;
			t = clock();
			for (int i = 0; i < n; i++) m = tri.getMasses(x);
			std::cout << "Clock: " << (clock() - t) << std::endl;
			std::cout << m << std::endl;
			
			const unsigned int numBlocks = 10000;
			const unsigned int numThreads = 512;
			std::cout << "Doing same kind of shit on the device(" << ((long long)numBlocks * (long long)numThreads * (long long)ITERATIONS) << " times(combined))..." << std::endl;
			t = clock();
			Vector3 *res;
			cudaMalloc(&res, sizeof(Vector3));
			testOnCuda<<<numBlocks, numThreads>>>(tri, x, ITERATIONS, res);
			cudaDeviceSynchronize();
			Vector3 r; if (cudaMemcpy(&r, res, sizeof(Vector3), cudaMemcpyDeviceToHost) != cudaSuccess) std::cout << "ERROR" << std::endl;
			cudaFree(res);
			std::cout << "Result: " << r << std::endl;
			std::cout << "Done. (" << (clock() - t) << "ms)" << std::endl;

			std::cout << "Enter any non-empty string to end test... ";
			char c; std::cin.get(c);
			int i = 0;
			while (true){
				std::cin.get(c);
				if (c == '\n') break;
				i++;
			}
			if (i > 0) break;
			std::cout << std::endl << std::endl;
		}
		std::cout << std::endl << std::endl;
	}
#undef MAX_COORD
#undef ITERATIONS
}
