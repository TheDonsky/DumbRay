#include"Tests.h"

namespace Tests {
	template<typename Function, typename... Args>
	inline static void call(std::string header, Function&& function, Args... args) {
		logDoubleLineBold();
		std::cout << header << std::endl;
		logLine();
		long long time = clock();
		function(args...);
		time = (clock() - time);
		logLineBold();
		std::cout << "Time: " << ((double)time) / CLOCKS_PER_SEC << std::endl << std::endl;
	}
	template<typename Function>
	inline static void runTest(Function&& function, std::string header) {
		logIntro();
		std::cout << header << std::endl;
		function();
		logOutro();
	}

	inline static void logPatern(std::string pattern, unsigned int numRepeats) {
		for (unsigned int i = 0; i < numRepeats; i++)
			std::cout << pattern;
		std::cout << std::endl;
	}

	inline static void logLine() {
		logPatern("-", 32);
	}
	inline static void logDoubleLine() {
		logPatern("-", 64);
	}
	inline static void logLineBold() {
		logPatern("=", 32);
	}
	inline static void logDoubleLineBold() {
		logPatern("=", 64);
	}

	inline static void logIntro() {
		logPatern("#", 76);
		logPatern("//\\\\", 19);
		logPatern("#", 76);
	}
	inline static void logOutro() {
		for (int i = 64; i > 0; i /= 2)
			logPatern("-", i);
		std::cout << std::endl << std::endl;
	}

}
