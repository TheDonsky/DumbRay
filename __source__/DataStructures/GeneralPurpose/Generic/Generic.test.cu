#include"Generic.test.cuh"
#include"Generic.cuh"
#include"../../Primitives/Pure/Vector3/Vector3.h"
#include"../../../Namespaces/Tests/Tests.h"

namespace GenericTest {
	namespace {
		namespace Private {
			struct FuncPack {
				void *unused;
				__dumb__ void clean() {
					unused = NULL;
				}
				template<typename Type>
				__dumb__ void use() {
					unused = NULL;
				}
			};

			__global__ static void valueCheckInt(Generic<FuncPack> *clone) {
				if (*clone->getObject<int>() != 73)
					printf("Esrror: expected %d, was %d\n", 73, *clone->getObject<int>());
			}

			__global__ static void valueCheckStacktor(Generic<FuncPack> *clone) {
				if (clone->getObject<Stacktor<int> >()->size() != 77773000)
					printf("Esrror: expected size was %d, got %d\n", 77773000, clone->getObject<Stacktor<int> >()->size());
			}

			void test() {
				bool pass = true;
				Generic<FuncPack> gen;
				Generic<FuncPack> genA;
				Generic<FuncPack> genB;
				gen.use<Vector3>(0.0f, 0.0f, 0.0f);
				if ((*gen.getObject<Vector3>()) != Vector3::zero()) {
					std::cout << "Value mismatch: was " << (*gen.getObject<Vector3>()) << ", expected " << Vector3::zero() << std::endl;
					pass = false;
				}

				genB = gen;
				if ((*genB.getObject<Vector3>()) != Vector3::zero()) {
					std::cout << "Value mismatch: was " << (*genB.getObject<Vector3>()) << ", expected " << Vector3::zero() << std::endl;
					pass = false;
				}

				gen = genA;
				if (gen.getObject<Vector3>() != NULL) {
					std::cout << "Value mismatch: was " << gen.getObject<Vector3>() << ", expected NULL" << std::endl;
					pass = false;
				}
				genA = genA;

				genB = genB;
				gen = genB;
				if ((*gen.getObject<Vector3>()) != Vector3::zero()) {
					std::cout << "Value mismatch: was " << (*gen.getObject<Vector3>()) << ", expected " << Vector3::zero() << std::endl;
					pass = false;
				}

				if (pass)
					std::cout << "Ok so far... Running kernel tests (you'll see tests in case of a failure)" << std::endl;

				genB.use<int>(73);
				Generic<FuncPack> *clone = genB.upload();
				if (clone == NULL) {
					std::cout << "Clone allocation failed...." << std::endl;
					return;
				}
				valueCheckInt << <1, 1 >> > (clone);
				cudaDeviceSynchronize();
				if (!Generic<FuncPack>::dispose(clone, 1)) {
					std::cout << "First dispose failed..." << std::endl;
					return;
				}
				Stacktor<int> stacktor;
				stacktor.flush(77773000);
				genB.use<Stacktor<int> >(stacktor);
				if (!genB.uploadAt(clone)) {
					std::cout << "Second upload failed" << std::endl;
					cudaFree(clone);
					return;
				}
				valueCheckStacktor << <1, 1 >> > (clone);
				cudaDeviceSynchronize();
				if (!Generic<FuncPack>::dispose(clone))
					std::cout << "Second dispose failed..." << std::endl;
				cudaFree(clone);
			}
		}
	}

	void test() {
		Tests::runTest(Private::test, "Running Generic test...");
	}
}