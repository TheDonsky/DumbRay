#include "ParserNetwork.h"
#include "Error.h"


namespace Parser {
#define SRC "ParserNetwork.cpp"

	ParserNetwork::ParserId ParserNetwork::Builder::addParser(const RecursiveParser &parser) {
		if (target == NULL) Error::fatal("Target missing", SRC, "ParserNetworkBuilder::addParser", Error::NULL_POINTER);
		size_t rv = target->parsers.size();
		target->parsers.push_back(parser);
		return rv;
	}
	void ParserNetwork::Builder::insert(ParserId subParser, ParserId parser) {
		if (target == NULL) Error::fatal("Target missing", SRC, "ParserNetworkBuilder::insert", Error::NULL_POINTER);
		if ((parser >= target->parsers.size()) || (subParser >= target->parsers.size()))
			Error::fatal("Invalid identifier(s)", SRC, "ParserNetworkBuilder::insert", Error::VALUE_ERROR);
		Instertion insertion;
		insertion.sub = subParser;
		insertion.super = parser;
		connections.push_back(insertion);
	}
	void ParserNetwork::Builder::selectMainParser(ParserId id) {
		if (target == NULL) Error::fatal("Target missing", SRC, "ParserNetworkBuilder::selectMainParser", Error::NULL_POINTER);
		if (id >= target->parsers.size()) Error::fatal("Invalid id", SRC, "ParserNetworkBuilder::selectMainParser", Error::VALUE_ERROR);
		mainIndex = id;
	}


	namespace {
		static void constructDefaultNetwork(ParserNetwork::Builder &builder) {
			ParserNetwork::ParserId defaultParser = builder.addParser(RecursiveParser());
			builder.selectMainParser(defaultParser);
		}
	}


	ParserNetwork::Builder::Builder(ParserNetwork::BuildFn buildFn) {
		if (buildFn == NULL) buildFn = constructDefaultNetwork;
		build = buildFn;
	}


	void ParserNetwork::Builder::buildNetwork(ParserNetwork *network) {
		if (network == NULL) Error::fatal("Null network pointer", SRC, "ParserNetworkBuilder::buildNetwork", Error::NULL_POINTER);
		network->parsers.clear();
		network->mainParser = NULL;
		target = network;
		mainIndex = 0;
		build(*this);
		for (size_t i = 0; i < connections.size(); i++)
			network->parsers[connections[i].super].addSubParser(&network->parsers[connections[i].sub]);
		network->mainParser = (&network->parsers[mainIndex]);
		connections.clear();
		target = NULL;
	}



	ParserNetwork::ParserNetwork(BuildFn builder) {
		Builder(builder).buildNetwork(this);
	}
	std::vector<RecursiveParser::Token> ParserNetwork::parse(const std::string &text)const {
		std::vector<RecursiveParser::Token> rv;
		parse(text, rv);
		return rv;
	}
	void ParserNetwork::parse(const std::string &text, std::vector<RecursiveParser::Token> &result)const {
		if (mainParser == NULL) Error::fatal("Main Parser missing", SRC, "ParserNetwork::parse(text, result)", Error::NULL_POINTER);
		else mainParser->parse(text, result);
	}

}
