#include"Tests.h"


template<typename Function, typename... Args>
static void Tests::call(std::string header, Function&& function, Args... args){
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
static void Tests::runTest(Function&& function, std::string header){
	logIntro();
	std::cout << header << std::endl;
	function();
	logOutro();
}

static void Tests::logPatern(std::string pattern, unsigned int numRepeats){
	for (unsigned int i = 0; i < numRepeats; i++)
		std::cout << pattern;
	std::cout << std::endl;
}

static void Tests::logLine(){
	logPatern("-", 32);
}
static void Tests::logDoubleLine(){
	logPatern("-", 64);
}
static void Tests::logLineBold(){
	logPatern("=", 32);
}
static void Tests::logDoubleLineBold(){
	logPatern("=", 64);
}

static void Tests::logIntro(){
	logPatern("#", 76);
	logPatern("//\\\\", 19);
	logPatern("#", 76);
}
static void Tests::logOutro(){
	for (int i = 64; i > 0; i /= 2)
		logPatern("-", i);
	std::cout << std::endl << std::endl;
}


