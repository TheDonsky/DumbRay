#pragma once
#include"Lense.cuh"
#include"../../../../Namespaces/Tests/Tests.h"
#include"../../../GeneralPurpose/Stacktor/Stacktor.cuh"


namespace LenseTest {
	namespace Private {
		class Garbage : public Stacktor<int> {
		public:
			__dumb__ Garbage() {
				flush(77773000);
			}

			__dumb__ void getScreenPhoton(const Vector2 &screenSpacePosition, PhotonPack &pack)const { 
#ifndef __CUDA_ARCH__
				printf("LenseTest::Private::Garbage::getScreenPhoton() called on HOST\n");
#else
				printf("LenseTest::Private::Garbage::getScreenPhoton() called on DEVICE\n");
#endif // !__CUDA_ARCH__
			}
			__dumb__ Photon toScreenSpace(const Photon &photon)const { return Photon(); }
			__dumb__ void getColor(const Vector2 &screenSpacePosition, Photon photon, Color &result)const { }
		};
	}
}

COPY_TYPE_TOOLS_IMPLEMENTATION(LenseTest::Private::Garbage, Stacktor<int>);

namespace LenseTest {
	namespace Private {

		__global__ static void valueCheckGarbage(Lense *clone) {
			printf("On kernel\n");
			if (clone->getObject<Garbage>()->size() != 77773000)
				printf("Error: expected size was %d, got %d\n", 77773000, clone->getObject<Garbage>()->size());
			else printf("Correct data on kernel\n");
			PhotonPack pack; 
			clone->getScreenPhoton(Vector2(0, 0), pack);
		}

		static void testMemory() {
			Lense lense;
			lense.use<Garbage>();
			PhotonPack pack;
			lense.getScreenPhoton(Vector2(0, 0), pack);
			Lense *clone = NULL;
			clone = lense.upload();
			if (clone == NULL) {
				std::cout << "Error occured while uploading" << std::endl;
				return;
			}
			std::cout << "Upload successful (if there are any errors, you'll se them below)" << std::endl;
			valueCheckGarbage<<<1, 1>>>(clone);
			cudaDeviceSynchronize();
			if (!Lense::dispose((Generic<LenseFunctionPack>*)clone)) {
				std::cout << "Error occured while disposing" << std::endl;
				return;
			}
			if (cudaFree(clone) != cudaSuccess)
				std::cout << "Error occured on deallocation attempt" << std::endl;


			Stacktor<Lense> lenses;
			lenses.flush(4);
			for (int i = 0; i < lenses.size(); i++)
				lenses[i].use<Garbage>();
			Stacktor<Lense> *lensesClone = lenses.upload();
			Stacktor<Lense>::dispose(lensesClone);
			cudaFree(lensesClone);
		}
	}

	void testMemory() {
		while (true) {
			Tests::runTest(Private::testMemory, "Running Lense memory test...");
			std::cout << "Enter anything to re-run the test: ";
			std::string s;
			std::getline(std::cin, s);
			if (s.length() <= 0) break;
		}
	}
}



