#pragma once
#include<iostream>
#include<string>
#include<time.h>



namespace Tests{
	template<typename Function, typename... Args>
	static void call(std::string header, Function&& function, Args... args);
	template<typename Function>
	static void runTest(Function&& function, std::string header);

	static void logPatern(std::string pattern, unsigned int numRepeats);

	static void logLine();
	static void logDoubleLine();
	static void logLineBold();
	static void logDoubleLineBold();
	
	static void logIntro();
	static void logOutro();
}





#include"Tests.impl.h"

