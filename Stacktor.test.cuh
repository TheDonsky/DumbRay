#pragma once

#include"Stacktor.cuh"
#include<iostream>
#include<string>
#include<time.h>
#include"Tests.h"


namespace StacktorTest{
	namespace Local{
		namespace Private{
			static void pushPerformance(){
				const int n = 50000000;
				std::cout << "Calling push() " << n << " times" << std::endl;
				Stacktor<Stacktor<int, 16>, 1> s;
				for (int i = 0; i < n; i++)
					s.push(Stacktor<int, 16>(i, i + 1, i + 2, i + 3));
			}
			static void operatorTest(){
				const int n = 500000000;
				std::cout << "Calling push() " << n << " times" << std::endl;

				Tests::logLine();
				std::cout << "Non constant reference: " << std::endl;
				Stacktor<int> s;
				for (int i = 0; i < n; i++) s.push(i);
				if (s.size() != n) std::cout << "ERROR: size does not match" << std::endl;
				else std::cout << "Size OK" << std::endl;
				int valueErrorCount = 0;
				for (int i = 0; i < s.size(); i++) if (s[i] != i) valueErrorCount++;
				std::cout << "operator[] value error count: " << valueErrorCount << std::endl;
				valueErrorCount = 0;
				for (int i = 0; i < s.size(); i++) if ((*(s + i)) != i) valueErrorCount++;
				std::cout << "operator+ value error count: " << valueErrorCount << std::endl;

				Tests::logLine();
				std::cout << "Constant reference: " << std::endl;
				const Stacktor<int>& cs = s;
				valueErrorCount = 0;
				for (int i = 0; i < s.size(); i++) if (cs[i] != i) valueErrorCount++;
				std::cout << "operator[] value error count: " << valueErrorCount << std::endl;
				valueErrorCount = 0;
				for (int i = 0; i < s.size(); i++) if ((*(cs + i)) != i) valueErrorCount++;
				std::cout << "operator+ value error count: " << valueErrorCount << std::endl;

				Tests::logLine();
				std::cout << "Clone: " << std::endl;
				Stacktor<int> csc(s);
				if (csc.size() != s.size()) std::cout << "ERROR: size mismatch" << std::endl;
				valueErrorCount = 0;
				for (int i = 0; i < s.size(); i++) if (csc[i] != i) valueErrorCount++;
				std::cout << "operator[] value error count: " << valueErrorCount << std::endl;
				valueErrorCount = 0;
				for (int i = 0; i < s.size(); i++) if ((*(csc + i)) != i) valueErrorCount++;
				std::cout << "operator+ value error count: " << valueErrorCount << std::endl;
			}

			static void checkVariadic(Stacktor<int> &s, int size){
				bool passed = true;
				if (s.size() != size){
					passed = false;
					std::cout << "Error: size mismatch" << std::endl;
				}
				for (int i = 0; i < s.size(); i++)
					if (s[i] != i + 1){
						passed = false;
						std::cout << "ERROR: value mismatch on index " << i << "; s[" << i << "] = " << s[i] << std::endl;
					}
				if (passed) std::cout << "PASSED" << std::endl;
				else std::cout << "FAILED" << std::endl;
			}

			static void variadicTest(){
				std::cout << "Size 4 Constructor: " << std::endl;
				Stacktor<int> s0(1, 2, 3, 4);
				checkVariadic(s0, 4);

				Tests::logLine();
				std::cout << "Size 1 Constructor: " << std::endl;
				Stacktor<int> s1(1);
				checkVariadic(s1, 1);

				Tests::logLine();
				std::cout << "Size 2 Constructor: " << std::endl;
				Stacktor<int> s2(1, 2);
				checkVariadic(s2, 2);

				Tests::logLine();
				std::cout << "Size 4 push: " << std::endl;
				Stacktor<int> s3;
				s3.push(1, 2, 3, 4);
				checkVariadic(s3, 4);

				Tests::logLine();
				std::cout << "Size 1 push: " << std::endl;
				Stacktor<int> s4;
				s4.push(1);
				checkVariadic(s4, 1);

				Tests::logLine();
				std::cout << "Size 2 push: " << std::endl;
				Stacktor<int> s5;
				s5.push(1, 2);
				checkVariadic(s5, 2);
			}

			static void popTest(){
				Stacktor<int> s(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
				const Stacktor<int>& cs = s;
				bool passed = true;
				if (s.peek() != 10){ passed = false; std::cout << "ERROR: incorrect peek" << std::endl; }
				if (cs.peek() != 10){ passed = false; std::cout << "ERROR: incorrect const peek" << std::endl; }
				if (s.pop() != 10){ passed = false; std::cout << "ERROR: incorrect pop" << std::endl; }
				if (cs.peek() != 9){ passed = false; std::cout << "ERROR: incorrect const peek after pop" << std::endl; }

				s.remove(2);
				for (int i = 0; i < s.size(); i++){
					int val = i + 1; if (i >= 2) val++;
					if (s[i] != val){ passed = false; std::cout << "ERROR: value mismatch" << std::endl; }
				}

				if (s.swapPop(4) != 6){ passed = false; std::cout << "ERROR: incorrect swapPop" << std::endl; }
				if (s.size() != 7){ passed = false; std::cout << "ERROR: size mismatch" << std::endl; }
				if (s[4] != 9 || s[5] != 7 || s[6] != 8){ passed = false; std::cout << "ERROR: value mismatch" << std::endl; }

				int n = s.size();
				for (int i = 0; i < n; i++){
					if (s.empty()){ passed = false; std::cout << "ERROR: empty too soon" << std::endl; }
					s.pop();
				}
				if (!s.empty()){ passed = false; std::cout << "ERROR: not empty after popping" << std::endl; }
				if (s.size() != 0){ passed = false; std::cout << "ERROR: size mismatch after clearing with pop" << std::endl; }

				s.push(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);
				if (s.size() != 10){ passed = false; std::cout << "ERROR: size mismatch after pushing" << std::endl; }
				s.clear();
				if (s.size() != 0 || (!s.empty())){ passed = false; std::cout << "ERROR: not empty after clearing" << std::endl; }

				if (passed) std::cout << "PASSED" << std::endl;
				else std::cout << "FAILED" << std::endl;
			}

			static void test(){
				Tests::call("Push performance: ", pushPerformance);
				Tests::call("Testing operators: ", operatorTest);
				Tests::call("Running variadic test", variadicTest);
				Tests::call("Running pop/peek/remove test", popTest);
			}
		}

		static void test(){
			Tests::runTest(Private::test, "Testing local functionality(host side)...");
		}
	}

	namespace Load{
		namespace Private{
			template<int stackDataSize>
			__global__ static void testUplodedIntegrity(Stacktor<int, stackDataSize> *s, int size, int count){
				int ind = StacktorPrivateKernels::getStartIndex();
				int end = StacktorPrivateKernels::getEndIndex(count);
				while (ind < end){
					Stacktor<int, stackDataSize> &input = s[ind];
					bool success = true;
					if (input.size() != size){
						printf("Size mismatch: %d != %d\n", input.size(), size);
						success = false;
					}
					for (int i = 0; i < input.size(); i++){
						if (input[i] != i){
							printf("Value mismatch: %d != %d\n", input[i], i);
							success = false;
						}
						if (i >= 256) break;
					}
					if (success){
						if (ind == 0) printf("Clone integrity test PASSED\n");
						else if (count <= 256) printf("Clone integrity test PASSED(id:%d)\n", ind);
					}
					else printf("Clone integrity test FAILED\n");
					ind++;
				}
			}

			template<int stackDataSize>
			static void testUpload(int size, int count){
				Tests::logLine();
				long timer = clock();
				Stacktor<int, stackDataSize> *s = new Stacktor<int, stackDataSize>[count];
				Stacktor<int, stackDataSize> *dev;
				if (cudaMalloc(&dev, sizeof(Stacktor<int, stackDataSize>) * count) != cudaSuccess){
					std::cout << "ERROR allocating memory for test" << std::endl;
					return;
				}
				std::cout << "Allocation clock: " << (clock() - timer) << std::endl; timer = clock();
				for (int j = 0; j < count; j++)
					for (int i = 0; i < size; i++) s[j].push(i);
				std::cout << "Fill clock: " << (clock() - timer) << std::endl; timer = clock();
				if (Stacktor<int, stackDataSize>::upload(s, dev, count)){
					std::cout << "Upload clock: " << (clock() - timer) << std::endl; timer = clock();
					int nBlocks = StacktorPrivateKernels::getBlockCount(count);
					int nthreads = StacktorPrivateKernels::getThreadCount();
					testUplodedIntegrity<<<nBlocks, nthreads>>>(dev, size, count);
					cudaDeviceSynchronize();
					std::cout << "Check clock: " << (clock() - timer) << std::endl; timer = clock();
				}
				else std::cout << "ERROR uploading for test" << std::endl;
				if (!Stacktor<int, stackDataSize>::dispose(dev, 1))
					std::cout << "ERROR disposing" << std::endl;
				else std::cout << "Dispose clock: " << (clock() - timer) << std::endl; timer = clock();
				cudaFree(dev);
				std::cout << "Clone free clock: " << (clock() - timer) << std::endl; timer = clock();
				delete[] s;
				std::cout << "Delete[] clock: " << (clock() - timer) << std::endl;
			}

			static void testSimple(){
				testUpload<32>(16, 1);
				testUpload<16>(16, 1);
				testUpload<8>(16, 1);
				typedef void(*voidFunc)(int, int);
				Tests::call("16 x 32024000 Upload", (voidFunc)(testUpload<16>), 32024000, 16);
				Tests::call("8024000 x 16 Upload", (voidFunc)(testUpload<16>), 16, 8024000);
				Tests::call("32024000 x 16 Upload", (voidFunc)(testUpload<16>), 16, 32024000);
			}

			typedef Stacktor<int, 4> Node;

			__global__ static void addToIndividuals(Stacktor<Node> *nodes){
				int ind = blockIdx.x * 256 + threadIdx.x;
				if (ind >= nodes->size()) return;
				(*nodes)[ind].push(1);
				if (threadIdx.x == 0)
					printf("block: %d; value: %d\n", blockIdx.x, (*nodes)[ind].top());
			}

			static void testComplex(){
				Stacktor<Node> nodes;
				for (int i = 0; i < 2048; i++){
					nodes.push(Node());
					for (int j = 0; j < 131071; j++)
						nodes.top().push(j);
				}
				Stacktor<Node> *dev; cudaMalloc(&dev, sizeof(Stacktor<Node>));
				if (!Stacktor<Node>::upload(&nodes, dev)) std::cout << "ERROR uploading" << std::endl;
				addToIndividuals<<<nodes.size() / 256, 256>>>(dev);
				cudaDeviceSynchronize();
				if (!Stacktor<Node>::dispose(dev)) std::cout << "ERROR disposing" << std::endl;
				cudaFree(dev);
			}

			typedef Stacktor<Node, 4> Stacktor2;
			typedef Stacktor<Stacktor2, 4> Stacktor3;

			__global__ static void changeOne(Stacktor3 *nodes){
				(*nodes)[0] = Stacktor2(
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
			}

			__global__ static void addToIndividuals(Stacktor3 *nodes){
				int ind = blockIdx.x * 32 + threadIdx.x;
				if (ind >= nodes->size()) return;
				(*nodes)[ind][0] = Node(1, 2, 3, 4, 5);
				(*nodes)[ind].push(Node());
				(*nodes)[ind].top().push(1, 2, 3, 4, 5);
			}

			__global__ static void addOne(Stacktor3 *nodes){
				(*nodes)[0] = Stacktor2(
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
					Node(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
				(*nodes).push(Stacktor2());
			}

			static void evenMoreComplexTest(){
				Stacktor3 s;
				for (int i = 0; i < 1024; i++){
					s.push(Stacktor2());
					for (int j = 0; j < 64; j++){
						s.top().push(Node());
						for (int k = 0; k < 256; k++)
							s.top().top().push(k);
					}
				}
				std::cout << "Load test started" << std::endl;
				Stacktor3 *dev; cudaMalloc(&dev, sizeof(Stacktor3));
				if (!Stacktor3::upload(&s, dev)) std::cout << "ERROR uploading" << std::endl;
				changeOne<<<1, 1>>>(dev);
				addToIndividuals<<<s.size() / 32, 32>>>(dev);
				if (cudaDeviceSynchronize() != cudaSuccess) std::cout << "Kernel error" << std::endl;
				if (!Stacktor3::dispose(dev)) std::cout << "ERROR disposing" << std::endl;
				if (!Stacktor3::upload(&s, dev)) std::cout << "ERROR uploading" << std::endl;
				addOne<<<1, 1>>>(dev);
				if (cudaDeviceSynchronize() != cudaSuccess) std::cout << "Kernel error" << std::endl;
				if (!Stacktor3::dispose(dev)) std::cout << "ERROR disposing" << std::endl;
				std::cout << "Load test ended" << std::endl;
				cudaFree(dev);
			}

			static void test(){
				Tests::call("Simple upload test", testSimple);
				Tests::call("Complex upload test", testComplex);
				Tests::call("Even more complex upload test", evenMoreComplexTest);
			}
		}

		static void test(){
			Tests::runTest(Private::test, "Testing load functionality...");
		}
	}

	static void fullTest(){
		Local::test();
		Load::test();
	}
}

