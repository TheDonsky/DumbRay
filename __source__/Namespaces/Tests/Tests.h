#pragma once
#include<iostream>
#include<string>
#include<time.h>



namespace Tests{
	template<typename Function, typename... Args>
	inline static void call(std::string header, Function&& function, Args... args);
	template<typename Function, typename... Args>
	inline static void runTest(Function&& function, std::string header, Args... args);

	inline static void logPatern(std::string pattern, unsigned int numRepeats);

	inline static void logLine();
	inline static void logDoubleLine();
	inline static void logLineBold();
	inline static void logDoubleLineBold();
	
	inline static void logIntro();
	inline static void logOutro();
}





#include"Tests.impl.h"

