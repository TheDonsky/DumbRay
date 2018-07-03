#pragma once
#include "RecursiveParser.h"
#define PARSER_NETWORK_PARSER_ID_TYPE size_t


namespace Parser {

	class ParserNetwork {
	public:
		class Builder;
		typedef void(*BuildFn)(Builder &builder);
		typedef size_t ParserId;

		class Builder {
		public:
			ParserId addParser(const RecursiveParser &parser);
			void insert(ParserId subParser, ParserId parser);
			void selectMainParser(ParserId id);

		private:
			friend class ParserNetwork;
			struct Instertion {
				size_t super, sub;
			};

			BuildFn build;
			ParserNetwork *target;
			size_t mainIndex;
			std::vector<Instertion> connections;

			Builder(BuildFn buildFn);
			inline Builder(const Builder &) {}
			inline Builder &operator=(const Builder &) { return (*this); }
			void buildNetwork(ParserNetwork *network);
		};


	public:
		ParserNetwork(BuildFn builder);
		std::vector<RecursiveParser::Token> parse(const std::string &text)const;
		void parse(const std::string &text, std::vector<RecursiveParser::Token> &result)const;

	private:
		friend class ParserNetworkBuilder;
		inline ParserNetwork(const ParserNetwork &) {}
		inline ParserNetwork &operator=(const ParserNetwork &) { return (*this); }
		std::vector<RecursiveParser> parsers;
		const RecursiveParser *mainParser;
	};


}
