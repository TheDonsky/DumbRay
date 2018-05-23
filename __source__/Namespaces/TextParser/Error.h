#pragma once
#include <string>


namespace Parser {

	namespace Error {
		enum Codes {
			NONE = 0,
			ALLOCATION_FALURE = 1,
			NULL_POINTER = 2,
			INVALID_POINTER = 3,
			VALUE_ERROR = 4
		};

		void log(const std::string &error, const std::string &file, const std::string &function);
		void fatal(const std::string &error, const std::string &file, const std::string &function, int errorCode);
	}

}

