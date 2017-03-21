#include"TypeTools.test.cuh"
#include"TypeTools.cuh"
#include"Stacktor.cuh"
#include"Vector3.h"
#include"Tests.h"


namespace TypeToolsTest {
	struct OneElem;
	struct TwoElem;
	struct ThreeElem;
	struct FourElem;
	template<typename Type> struct OneElemTemplate;
	template<typename Type> struct TwoElemTemplate;
	template<typename Type> struct ThreeElemTemplate;
	template<typename Type> struct FourElemTemplate;
	class SomethingTerriblyHuge;
}
TYPE_TOOLS_REDEFINE_1_PART(TypeToolsTest::OneElem, int);
TYPE_TOOLS_REDEFINE_2_PART(TypeToolsTest::TwoElem, int, Vector3);
TYPE_TOOLS_REDEFINE_3_PART(TypeToolsTest::ThreeElem, int, Vector3, Stacktor<char>);
TYPE_TOOLS_REDEFINE_4_PART(TypeToolsTest::FourElem, int, Vector3, Stacktor<char>, Stacktor<Stacktor<short> >);
TYPE_TOOLS_REDEFINE_1_PART_TEMPLATE(TypeToolsTest::OneElemTemplate, TemplateType, typename TemplateType);
TYPE_TOOLS_REDEFINE_2_PART_TEMPLATE(TypeToolsTest::TwoElemTemplate, TemplateType, TemplateType, typename TemplateType);
TYPE_TOOLS_REDEFINE_3_PART_TEMPLATE(TypeToolsTest::ThreeElemTemplate, TemplateType, TemplateType, TemplateType, typename TemplateType);
TYPE_TOOLS_REDEFINE_4_PART_TEMPLATE(TypeToolsTest::FourElemTemplate, TemplateType, TemplateType, TemplateType, TemplateType, typename TemplateType);
TYPE_TOOLS_REDEFINE_1_PART(TypeToolsTest::SomethingTerriblyHuge, Stacktor<int>);
namespace TypeToolsTest {
	struct OneElem {
		int elem;
		__dumb__ void setValues() {
			elem = 9978;
		}
		__dumb__ bool checkValues() const {
			return (elem == 9978);
		}
		TYPE_TOOLS_ADD_COMPONENT_GETTER(OneElem, elem);
	};
	struct TwoElem {
		int elem0;
		Vector3 elem1;
		__dumb__ void setValues() {
			elem0 = 9978;
			elem1 = Vector3(213.1f, 213.324f, 29130.28f);
		}
		__dumb__ bool checkValues() const {
			return (elem0 == 9978 && elem1 == Vector3(213.1f, 213.324f, 29130.28f));
		}
		TYPE_TOOLS_ADD_COMPONENT_GETTERS_2(TwoElem, elem0, elem1);
	};
	struct ThreeElem {
		int elem0;
		Vector3 elem1;
		Stacktor<char> elem2;
		__dumb__ void setValues() {
			elem0 = 9978;
			elem1 = Vector3(213.1f, 213.324f, 29130.28f);
			elem2.clear();
			elem2.push(0, 1, 2, 3, 4);
		}
		__dumb__ bool checkValues() const {
			bool stacktorOk = (elem2.size() == 5);
			if (stacktorOk)
				for (int i = 0; i < 5; i++)
					stacktorOk &= (elem2[i] == i);
			return (elem0 == 9978 && elem1 == Vector3(213.1f, 213.324f, 29130.28f) && stacktorOk);
		}
		TYPE_TOOLS_ADD_COMPONENT_GETTERS_3(ThreeElem, elem0, elem1, elem2);
	};
	struct FourElem {
		int elem0;
		Vector3 elem1;
		Stacktor<char> elem2;
		Stacktor<Stacktor<short> > elem3;
		__dumb__ void setValues() {
			elem0 = 9978;
			elem1 = Vector3(213.1f, 213.324f, 29130.28f);
			elem2.clear();
			elem2.push(0, 1, 2, 3, 4);
			elem3.clear();
			elem3.push(Stacktor<short>(9, 8, 7), Stacktor<short>(5), Stacktor<short>());
		}
		__dumb__ bool checkValues() const {
			bool stacktorOk = (elem2.size() == 5);
			if (stacktorOk)
				for (int i = 0; i < 5; i++)
					stacktorOk &= (elem2[i] == i);
			stacktorOk &= (elem3.size() == 3);
			if (stacktorOk) stacktorOk &= (elem3[0].size() == 3 && elem3[1].size() == 1 && elem3[2].size() == 0);
			if (stacktorOk) stacktorOk &= (elem3[0][0] == 9 && elem3[0][1] == 8 && elem3[0][2] == 7 && elem3[1][0] == 5);
			return (elem0 == 9978 && elem1 == Vector3(213.1f, 213.324f, 29130.28f) && stacktorOk);
		}
		TYPE_TOOLS_ADD_COMPONENT_GETTERS_4(FourElem, elem0, elem1, elem2, elem3);
	};
	template<typename Type> 
	struct OneElemTemplate {
		Type elem;
		__dumb__ void setValues() { elem.setValues(); }
		__dumb__ bool checkValues() const { return elem.checkValues(); }
		TYPE_TOOLS_ADD_COMPONENT_GETTER(OneElemTemplate, elem);
	};
	template<typename Type> 
	struct TwoElemTemplate {
		Type elem0, elem1;
		__dumb__ void setValues() { elem0.setValues(); elem1.setValues(); }
		__dumb__ bool checkValues() const { return (elem0.checkValues() && elem1.checkValues()); }
		TYPE_TOOLS_ADD_COMPONENT_GETTERS_2(TwoElemTemplate, elem0, elem1);
	};
	template<typename Type> 
	struct ThreeElemTemplate {
		Type elem0, elem1, elem2;
		__dumb__ void setValues() { elem0.setValues(); elem1.setValues(); elem2.setValues(); }
		__dumb__ bool checkValues() const { return (elem0.checkValues() && elem1.checkValues() && elem2.checkValues()); }
		TYPE_TOOLS_ADD_COMPONENT_GETTERS_3(ThreeElemTemplate, elem0, elem1, elem2);
	};
	template<typename Type> 
	struct FourElemTemplate {
		Type elem0, elem1, elem2, elem3;
		__dumb__ void setValues() { elem0.setValues(); elem1.setValues(); elem2.setValues(); elem3.setValues(); }
		__dumb__ bool checkValues() const { return (elem0.checkValues() && elem1.checkValues() && elem2.checkValues() && elem3.checkValues()); }
		TYPE_TOOLS_ADD_COMPONENT_GETTERS_4(FourElemTemplate, elem0, elem1, elem2, elem3);
	};
	class SomethingTerriblyHuge {
	private:
		Stacktor<int> data;
		TYPE_TOOLS_ADD_COMPONENT_GETTER(SomethingTerriblyHuge, data);
	public:
		__dumb__ void setValues() {
			data.clear();
			for (int i = 0; i < (1 << 28); i++)
				data.push(i);
			printf("PUSHED %d NUMBERS\n", (1 << 28));
		}
		__dumb__ bool checkValues() const {
			int size = data.size();
			if (size == (1 << 28)) {
				printf("DATA SIZE CORECT\n");
				printf("FOR SAKE OF PRESERVING TIME, CHECKING ONLY THE FIRST 256 VALUES\n");
				bool pass = true;
				for (int i = 0; i < 256; i++)
					pass &= (data[i] == i);
				return pass;
			}
			else {
				printf("DATA SIZE INCORRECT (%d INSTEAD OF %d)\n", size, (1 << 28));
				return false;
			}
		}
	};
}
TYPE_TOOLS_IMPLEMENT_1_PART(TypeToolsTest::OneElem);
TYPE_TOOLS_IMPLEMENT_2_PART(TypeToolsTest::TwoElem);
TYPE_TOOLS_IMPLEMENT_3_PART(TypeToolsTest::ThreeElem);
TYPE_TOOLS_IMPLEMENT_4_PART(TypeToolsTest::FourElem);
TYPE_TOOLS_IMPLEMENT_1_PART_TEMPLATE(TypeToolsTest::OneElemTemplate, typename TemplateType);
TYPE_TOOLS_IMPLEMENT_2_PART_TEMPLATE(TypeToolsTest::TwoElemTemplate, typename TemplateType);
TYPE_TOOLS_IMPLEMENT_3_PART_TEMPLATE(TypeToolsTest::ThreeElemTemplate, typename TemplateType);
TYPE_TOOLS_IMPLEMENT_4_PART_TEMPLATE(TypeToolsTest::FourElemTemplate, typename TemplateType);
TYPE_TOOLS_IMPLEMENT_1_PART(TypeToolsTest::SomethingTerriblyHuge);
namespace TypeToolsTest {
	template<typename Type>
	__global__ static void checkValues(const Type *object, bool *rv) {
		if (threadIdx.x == 0 && blockIdx.x == 0) {
			(*rv) = object->checkValues();
			if (*rv) printf("VALUES ARE CORRECT\n");
			else printf("VALUES ARE WRONG\n");
		}
	}
	template<typename Type>
	inline static bool testType(const std::string &className, bool log = true) {
		std::cout << "______________________________________" << std::endl;
		std::cout << "...........TEST STRATED..............." << std::endl;
		std::cout << "TESTING: " << className << std::endl;
		std::cout << "--------------------------" << std::endl;
		char rawMemory[sizeof(Type)];
		Type *elem = ((Type*)rawMemory);
		Type other;
		TypeTools<Type>::init(*elem);
		elem->setValues();
		TypeTools<Type>::transfer(*elem, other);
		TypeTools<Type>::swap(*elem, other);
		char hosCloneRaw[sizeof(Type)];
		Type *hosClone = ((Type*)hosCloneRaw);
		Type *devTarget;
		bool success = true;
		if (cudaMalloc((void**)&devTarget, sizeof(Type)) == cudaSuccess) {
			if (log) std::cout << "ALLOCATION SUCCESSFUL" << std::endl;
			if (TypeTools<Type>::prepareForCpyLoad(elem, hosClone, devTarget, 1)) {
				if (log) std::cout << "CPY LOAD PREPARATION SUCCESSFUL" << std::endl;
				if (cudaMemcpy(devTarget, hosClone, sizeof(Type), cudaMemcpyHostToDevice) == cudaSuccess) {
					if (log) std::cout << "UPLOAD SUCCESSFUL" << std::endl;
					bool *devRv;
					if (cudaMalloc(&devRv, sizeof(bool)) == cudaSuccess) {
						checkValues << <1, 1 >> > (devTarget, devRv);
						if (cudaDeviceSynchronize() == cudaSuccess) {
							if (log) std::cout << "KERNEL EXECUTION COMPLETE" << std::endl;
							bool kernelRv = false;
							if (cudaMemcpy(&kernelRv, devRv, sizeof(bool), cudaMemcpyDeviceToHost) != cudaSuccess) {
								std::cout << "FAILED LOADING RV" << std::endl;
								kernelRv = false;
							}
							success &= kernelRv;
							if (cudaDeviceSynchronize() != cudaSuccess) {
								std::cout << "DEVICE SYNCHRONISATION FAILED" << std::endl;
								success = false;
							}
						}
						else {
							std::cout << "KERNEL EXECUTION FAILED" << std::endl;
							success = false;
						}
						if (cudaFree(devRv) != cudaSuccess) {
							std::cout << "RETURN VALUE DEALLOCATION FAILED" << std::endl;
							success = false;
						}
					}
					else {
						std::cout << "FAILED TO ALLOCATE KERNEL RETURN VALUE";
						success = false;
					}
					if (TypeTools<Type>::devArrayNeedsToBeDisposed()) {
						if (TypeTools<Type>::disposeDevArray(devTarget, 1)) {
							if (log) std::cout << "DISPOSE SUCCESSFUL" << std::endl;
						}
						else {
							std::cout << "DISPOSE ERROR" << std::endl;
							success = false;
						}
					}
					else if (log) std::cout << "NO DISPOSAL NEEDED" << std::endl;
				}
				else {
					std::cout << "UPLOAD FAILED" << std::endl;
					success = false;
				}
			}
			else {
				std::cout << "UPLOAD ERROR" << std::endl;
				success = false;
			}
			if (cudaFree(devTarget) == cudaSuccess) {
				if (log) std::cout << "DEALLOCATION SUCCESSFUL" << std::endl;
			}
			else {
				std::cout << "DEALLOCATION ERROR" << std::endl;
				success = false;
			}
		}
		else {
			std::cout << "ALLOCATION ERROR" << std::endl;
			success = false;
		}
		TypeTools<Type>::dispose(*elem);
		std::cout << "...........TEST FINISHED............." << std::endl << std::endl;
		std::cout << "_______STATUS: " << (success ? "PASS" : "FAIL") << std::endl;
		return success;
	}
	static void testFunction() {
		int device;
		if (cudaGetDevice(&device) == cudaSuccess) {
			bool success = testType<OneElem>("OneElem");
			success &= testType<TwoElem>("TwoElem");
			success &= testType<ThreeElem>("ThreeElem");
			success &= testType<FourElem>("FourElem");
			success &= testType<OneElemTemplate<OneElem> >("OneElemTemplate<OneElem>");
			success &= testType<TwoElemTemplate<TwoElem> >("TwoElemTemplate<TwoElem>");
			success &= testType<ThreeElemTemplate<ThreeElem> >("ThreeElemTemplate<ThreeElem>");
			success &= testType<FourElemTemplate<FourElem> >("FourElemTemplate<FourElem>");
			success &= testType<SomethingTerriblyHuge>("SomethingTerriblyHuge");
			const int n = 8;
			std::cout << std::endl << std::endl << std::endl << "PRESS ENTER TO RE-RUN THE TEST FOR SomethingTerriblyHuge " << n << " MORE TIMES... ";
			std::string s;
			std::getline(std::cin, s);
			for (int i = 0; i < n; i++)
				success &= testType<SomethingTerriblyHuge>("SomethingTerriblyHuge", false);
			std::cout << std::endl << std::endl << std::endl << "============================================================" << std::endl;
			std::cout << "DONE; FULL TEST RESULT: " << (success ? "PASS" : "FAIL") << std::endl;
			std::cout << "MAKE SURE, RAM AND VRAM USAGES ARE UNDER CONTROLL..." << std::endl;
		}
		else std::cout << "NO ACTIVE CUDA DEVICE FOUND TO RUN THE TEST..." << std::endl;
	}
	void test() {
		while (true) {
			std::cout << "Enter anthing to run TypeTools test: ";
			std::string s;
			std::getline(std::cin, s);
			if (s.length() <= 0) break;
			Tests::runTest(testFunction, "RUNNING TESTS FOR DEFAULT IMPLEMENTATIONS OF TYPE_TOOLS");
		}
	}
}




