#include "Error.h"
#include <iostream>



namespace Parser {


	namespace Error {
		void log(const std::string &error, const std::string &file, const std::string &function) {
			std::cout << "ERROR: " << error << " at " << file << " - " << function << std::endl;
		}
		void fatal(const std::string &error, const std::string &file, const std::string &function, int errorCode) {
			log(error, file, function);
			system("PAUSE");
			exit(errorCode);
		}
	}

}
