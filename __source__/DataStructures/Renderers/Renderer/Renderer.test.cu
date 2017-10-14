#include "Renderer.test.cuh"
#include "Renderer.cuh"
#include "../../../Namespaces/Tests/Tests.h"
#include <time.h>



namespace RendererTest {
	namespace {
		class DummyRenderer : public Renderer {
		public:
			DummyRenderer(const ThreadConfiguration &configuration, bool verbose, int counter);
			virtual ~DummyRenderer();

		protected:
			virtual bool setupSharedData(const Info &info, void *& sharedData);
			virtual bool setupData(const Info &info, void *& data);
			virtual bool prepareIteration();
			virtual void iterateCPU(const Info &info);
			virtual void iterateGPU(const Info &info);
			virtual bool completeIteration();
			virtual bool clearData(const Info &info, void *& data);
			virtual bool clearSharedData(const Info &info, void *& sharedData);

		private:
			std::mutex printLock;
			bool dumpCalls;
			int cnt;

		public:
			int *total[128];
		};
		DummyRenderer::DummyRenderer(const ThreadConfiguration &configuration, bool verbose, int counter) : Renderer(configuration){
			dumpCalls = verbose;
			cnt = counter;
			for (int i = 0; i < 128; i++) total[i] = NULL;
		}
		DummyRenderer::~DummyRenderer() { 
			killRenderThreads(); 
		}
		bool DummyRenderer::setupSharedData(const Info &info, void *& sharedData) {
			if (dumpCalls) {
				printLock.lock();
				std::cout << "setupSharedData - ";
				if (info.isGPU()) std::cout << "GPU " << info.device << " ";
				else std::cout << "CPU ";
				std::cout << info.deviceThreadId << std::endl;
				printLock.unlock();
			}
			sharedData = (void*)(&info.device);
			return true; 
		}
		bool DummyRenderer::setupData(const Info &info, void *& data) {
			if (*(((int*)info.sharedData)) != info.device) std::cout << "ERROR: sharedData VALUE INCORRECT" << std::endl;
			data = (void*)(&info.deviceThreadId);
			if (dumpCalls) {
				printLock.lock(); 
				std::cout << "setupData - ";
				if (info.isGPU()) std::cout << "GPU " << info.device << " ";
				else std::cout << "CPU ";
				std::cout << info.deviceThreadId << std::endl;
				printLock.unlock();
			}
			total[info.globalThreadId] = new int;
			return true; 
		}
		bool DummyRenderer::prepareIteration() {
			if (dumpCalls) {
				printLock.lock();
				std::cout << "prepareIteration..." << std::endl;
				printLock.unlock();
			}
			return true;
		}
		void DummyRenderer::iterateCPU(const Info &info) {
			if (*(((int*)info.sharedData)) != info.device) std::cout << "ERROR: sharedData VALUE INCORRECT" << std::endl;
			if (*(((int*)info.data)) != info.deviceThreadId) std::cout << "ERROR: data VALUE INCORRECT" << std::endl;
			if (dumpCalls) {
				printLock.lock();
				std::cout << "iterateCPU - ";
				if (info.isGPU()) std::cout << "GPU " << info.device << " ";
				else std::cout << "CPU ";
				std::cout << info.deviceThreadId << std::endl;
				printLock.unlock();
			}
			int &v = (*total[info.globalThreadId]);
			for (int i = 0; i < cnt; i++) v++;
		}
		void DummyRenderer::iterateGPU(const Info &info) {
			if (*(((int*)info.sharedData)) != info.device) std::cout << "ERROR: sharedData VALUE INCORRECT" << std::endl;
			if (*(((int*)info.data)) != info.deviceThreadId) std::cout << "ERROR: data VALUE INCORRECT" << std::endl;
			if (dumpCalls) {
				printLock.lock();
				std::cout << "iterateGPU - ";
				if (info.isGPU()) std::cout << "GPU " << info.device << " ";
				else std::cout << "CPU ";
				std::cout << info.deviceThreadId << std::endl;
				printLock.unlock();
			}
			int &v = (*total[info.globalThreadId]);
			for (int i = 0; i < cnt; i++) v++;
		}
		bool DummyRenderer::completeIteration() {
			if (dumpCalls) {
				printLock.lock();
				std::cout << "completeIteration..." << std::endl;
				printLock.unlock();
			}
			return true;
		}
		bool DummyRenderer::clearData(const Info &info, void *& data) {
			if (*(((int*)info.sharedData)) != info.device) std::cout << "ERROR: sharedData VALUE INCORRECT" << std::endl;
			data = NULL;
			if (dumpCalls) {
				printLock.lock();
				std::cout << "clearData - ";
				if (info.isGPU()) std::cout << "GPU " << info.device << " ";
				else std::cout << "CPU ";
				std::cout << info.deviceThreadId << std::endl;
				printLock.unlock();
			}
			delete total[info.globalThreadId];
			return true; 
		}
		bool DummyRenderer::clearSharedData(const Info &info, void *& sharedData) {
			if (*(((int*)info.sharedData)) != info.device) std::cout << "ERROR: sharedData VALUE INCORRECT" << std::endl;
			sharedData = NULL;
			if (info.data != NULL) std::cout << "ERROR: data NOT NULL";
			if (dumpCalls) {
				printLock.lock();
				std::cout << "clearSharedData - ";
				if (info.isGPU()) std::cout << "GPU " << info.device << " ";
				else std::cout << "CPU ";
				std::cout << info.deviceThreadId << std::endl;
				printLock.unlock();
			}
			return true; 
		}

		void makeAndDestroy() {
			std::cout << std::endl << "--> FAST CREATE & DESTROY TEST: " << std::endl;
			Renderer::ThreadConfiguration configuration;
			DummyRenderer renderer(configuration, true, 0);
			renderer.iterate();
		}

		void testIterationsSpeed() {
			const int n = 8192;
			const int weight = 262144;
			Renderer::ThreadConfiguration configuration;
			DummyRenderer renderer(configuration, false, weight);
			std::cout << std::endl << "--> RUNNING " << n << " ITERATIONS (weight "<< weight << ")...." << std::endl;
			long t = clock();
			for (int i = 0; i < n; i++) renderer.iterate();
			long deltaTime = (clock() - t);
			float secs = (((float)deltaTime) / CLOCKS_PER_SEC);
			std::cout << "TIME: " << deltaTime << " CLOCK TICKS (" << secs << " sec)" << std::endl;
			std::cout << "SPEED: " << (((float)n) / secs) << " ITERATIONS PER SECOND" << std::endl << std::endl;
			int total = 0;
			for (int i = 0; i < 128; i++) 
				if(renderer.total[i] != NULL) total += (*renderer.total[i]);
			std::cout << "TOTAL: " << total << std::endl;
		}

		void runTest() {
			makeAndDestroy();
			testIterationsSpeed();
		}

		void testRenderer(){
			while (true) {
				std::cout << "Enter anthing to run basic Renderer template test: ";
				std::string s;
				std::getline(std::cin, s);
				if (s.length() <= 0) break;
				runTest();
			}
		}
	}

	void test() {
		Tests::runTest(testRenderer, "Testing basic Renderer template");
		cudaSetDevice(0);
	}
}

