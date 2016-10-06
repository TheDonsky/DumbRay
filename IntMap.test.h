#pragma once

#include"IntMap.h"
#include"Tests.h"



namespace IntMapTest{
	namespace Private{
		inline static void test(){
			IntMap<int> map;
			bool success = true;
			const int n = 1024000;
			for (int i = -n; i < n; i++)
				map[i] = i + 32;
			for (int i = -n; i < n; i++){
				if (!map.contains(i)){
					std::cout << "ERROR: VALUE NOT FOUND" << std::endl;
					success = false;
				}
				else if (map[i] != i + 32){
					std::cout << "ERROR: INVALID VALUE: " << i << " - " << map[i] << std::endl;
					success = false;
				}
			}
			if (success) std::cout << "PASSED" << std::endl;
			else  std::cout << "FAILED" << std::endl;
		}
	}

	inline static void test(){
		Tests::call("Running IntMap test", Private::test);
	}
}
