#include "Dson.h"
#include "../TextParser/ParserNetwork.h"
#include<sstream>
#include<iostream>
#include<new>



namespace Dson {
	namespace {
		typedef std::unordered_map<char, std::string> EscapeMap;
		static EscapeMap getEscapeMap() {
			std::unordered_map<char, std::string> result;
			result['"'] = "\\\"";
			result['\\'] = "\\\\";
			result['/'] = "\\/";
			result['\b'] = "\\b";
			result['\f'] = "\\f";
			result['\n'] = "\\n";
			result['\r'] = "\\r";
			result['\t'] = "\\t";
			return result;
		}
		static const EscapeMap escapeMap = getEscapeMap();

		static std::string stringToJsonSourceString(const std::string &text) {
			std::stringstream stream;
			stream << '"';
			for (size_t i = 0; i < text.size(); i++) {
				const char symbol = text[i];
				EscapeMap::const_iterator it = escapeMap.find(symbol);
				if (it != escapeMap.end()) stream << (it->second);
				else stream << symbol;
			}
			stream << '"';
			return stream.str();
		}
	}


	Object::~Object() { }





	Dict::Dict() {}
	Dict::Dict(const Dict &other) { copyFrom(other); }
	Dict &Dict::operator=(const Dict &other) { copyFrom(other); return (*this); }
	void Dict::copyFrom(const Dict &other) {
		if (this == (&other)) return;
		this->~Dict();
		new (this) Dict();
		for (std::unordered_map<std::string, Object*>::const_iterator it = other.map.begin(); it != other.map.end(); it++) {
			if (it->second == NULL) map[it->first] = NULL;
			else set(it->first, *it->second);
		}
	}
	Dict::Dict(Dict &&other) { stealFrom(other); }
	Dict &Dict::operator=(Dict &&other) { stealFrom(other); return (*this); }
	void Dict::stealFrom(Dict &other) {
		if (this == (&other)) return;
		this->~Dict();
		new (this) Dict();
		map = other.map;
		other.map = std::unordered_map<std::string, Object*>();
	}
	Dict::~Dict() {
		for (std::unordered_map<std::string, Object*>::iterator it = map.begin(); it != map.end(); it++)
			if (it->second != NULL) delete it->second;
	}
	size_t Dict::size()const { return map.size(); }
	bool Dict::contains(const std::string &key)const { return (map.find(key) != map.end()); }
	const Object& Dict::get(const std::string &key)const { return (*map.find(key)->second); }
	const Object& Dict::operator[](const std::string &key)const { return (*map.find(key)->second); }
	void Dict::set(const std::string &key, const Object &value) {
		remove(key);
		if (value.type() == DSON_DICT) map[key] = new Dict(*((Dict*)(&value)));
		else if (value.type() == DSON_ARRAY) map[key] = new Array(*((Array*)(&value)));
		else if (value.type() == DSON_STRING) map[key] = new String(*((String*)(&value)));
		else if (value.type() == DSON_NUMBER) map[key] = new Number(*((Number*)(&value)));
		else if (value.type() == DSON_BOOL) map[key] = new Bool(*((Bool*)(&value)));
		else if (value.type() == DSON_NULL) map[key] = new Null(*((Null*)(&value)));
	}
	void Dict::remove(const std::string &key) {
		std::unordered_map<std::string, Object*>::const_iterator it = map.find(key);
		if (it != map.end()) {
			delete it->second;
			map.erase(it);
		}
	}
	Dict::Type Dict::type()const { return DSON_DICT; }
	std::string Dict::toString(const std::string &baseInset, const std::string &inset)const {
		std::stringstream stream;
		stream << '{';
		if (size() > 0) {
			std::unordered_map<std::string, Object*>::const_iterator it = map.begin();
			std::string elemInset = (inset + baseInset);
			if (size() > 1) stream << "\n" << elemInset;
			while (true) {
				stream << stringToJsonSourceString(it->first) << ": " << ((it->second == NULL) ? "null" : it->second->toString(baseInset, elemInset));
				it++;
				if (it == map.end()) break;
				stream << ",\n" << elemInset;
			}
			if (size() > 1) stream << "\n" << inset;
		}
		stream << '}';
		return stream.str();
	}





	Array::Array() {}
	Array::Array(const Array &other) { copyFrom(other); }
	Array &Array::operator=(const Array &other) { copyFrom(other); return (*this); }
	void Array::copyFrom(const Array &other) {
		if (this == (&other)) return;
		this->~Array();
		new (this) Array();
		while (size() < other.size()) objects.push_back(NULL);
		for (size_t i = 0; i < other.objects.size(); i++)
			if (other.objects[i] != NULL) set(i, other[i]);
	}
	Array::Array(Array &&other) { stealFrom(other); }
	Array &Array::operator=(Array &&other) { stealFrom(other); return (*this); }
	void Array::stealFrom(Array &other) {
		if (this == (&other)) return;
		this->~Array();
		new (this) Array();
		objects = other.objects;
		other.objects = std::vector<Object*>();
	}
	Array::~Array() {
		for (size_t i = 0; i < objects.size(); i++)
			if (objects[i] != NULL) delete objects[i];
	}
	size_t Array::size()const { return objects.size(); }
	const Object& Array::get(size_t index)const { return (*objects[index]); }
	const Object& Array::operator[](size_t index)const { return (*objects[index]); }
	void Array::set(size_t index, const Object &value) {
		while (size() <= index) objects.push_back(NULL);
		if (objects[index] != NULL) delete objects[index];
		if (value.type() == DSON_DICT) objects[index] = new Dict(*((Dict*)(&value)));
		else if (value.type() == DSON_ARRAY) objects[index] = new Array(*((Array*)(&value)));
		else if (value.type() == DSON_STRING) objects[index] = new String(*((String*)(&value)));
		else if (value.type() == DSON_NUMBER) objects[index] = new Number(*((Number*)(&value)));
		else if (value.type() == DSON_BOOL) objects[index] = new Bool(*((Bool*)(&value)));
		else if (value.type() == DSON_NULL) objects[index] = new Null(*((Null*)(&value)));
	}
	void Array::push(const Object &value) { set(size(), value); }
	void Array::pop() {
		if (objects[size() - 1] != NULL) delete objects[size() - 1];
		objects.pop_back();
	}
	Array::Type Array::type()const { return DSON_ARRAY; }
	std::string Array::toString(const std::string &baseInset, const std::string &inset)const {
		std::stringstream stream;
		stream << '[';
		if (size() > 0) {
			size_t index = 0;
			std::string elemInset = (inset + baseInset);
			if (size() > 1) stream << "\n" << elemInset;
			while (true) {
				stream << ((objects[index] == NULL) ? "null" : objects[index]->toString(baseInset, elemInset));
				index++;
				if (index >= size()) break;
				stream << ",\n" << elemInset;
			}
			if (size() > 1) stream << "\n" << inset;
		}
		stream << ']';
		return stream.str();
	}





	std::string &String::value() { return val; }
	const std::string &String::value()const { return val; }
	String::Type String::type()const { return DSON_STRING; }
	std::string String::toString(const std::string &, const std::string &)const {
		return stringToJsonSourceString(val);
	}
	




	int Number::intValue()const { return ((int)val); }
	float Number::floatValue()const { return ((float)val); }
	double& Number::value() { return val; }
	double Number::value()const { return val; }
	Number::Type Number::type()const { return DSON_NUMBER; }
	std::string Number::toString(const std::string &, const std::string &)const {
		std::stringstream stream;
		stream << val;
		return stream.str();
	}





	bool &Bool::value() { return val; }
	bool Bool::value()const { return val; }
	Bool::Type Bool::type()const { return DSON_BOOL; }
	std::string Bool::toString(const std::string &, const std::string &)const { return (val ? "true" : "false"); }
	




	Null::Type Null::type()const { return DSON_NULL; }
	std::string Null::toString(const std::string &, const std::string &)const { return "null"; }





	namespace {
		static void buildDsonParserNetwork(Parser::ParserNetwork::Builder &builder) {
			Parser::RecursiveParser listParser;

			listParser.addEscapeSequence("  ", " ", Parser::RecursiveParser::ESCAPE_RECURSIVE);
			listParser.addEscapeSequence("\f", " ", Parser::RecursiveParser::ESCAPE_RECURSIVE);
			listParser.addEscapeSequence("\n", " ", Parser::RecursiveParser::ESCAPE_RECURSIVE);
			listParser.addEscapeSequence("\r", " ", Parser::RecursiveParser::ESCAPE_RECURSIVE);
			listParser.addEscapeSequence("\t", " ", Parser::RecursiveParser::ESCAPE_RECURSIVE);
			listParser.addEscapeSequence("\v", " ", Parser::RecursiveParser::ESCAPE_RECURSIVE);
			listParser.addEscapeSequence("  ", " ", Parser::RecursiveParser::ESCAPE_RECURSIVE);

			listParser.addEscapeSequence("{", " { ", Parser::RecursiveParser::ESCAPE_REPLACE);
			listParser.addEscapeSequence("}", " } ", Parser::RecursiveParser::ESCAPE_REPLACE);
			listParser.addEscapeSequence("[", " [ ", Parser::RecursiveParser::ESCAPE_REPLACE);
			listParser.addEscapeSequence("]", " ] ", Parser::RecursiveParser::ESCAPE_REPLACE);
			listParser.addEscapeSequence(",", " , ", Parser::RecursiveParser::ESCAPE_REPLACE);
			listParser.addEscapeSequence(":", " : ", Parser::RecursiveParser::ESCAPE_REPLACE);

			listParser.addDelimiter(" ");

			Parser::ParserNetwork::ParserId listParserId = builder.addParser(listParser);

			Parser::ParserNetwork::ParserId dictParserId;
			{
				Parser::RecursiveParser dictParser(listParser);
				dictParser.addEntryPoint("{", false);
				dictParser.addExitPoint("}", false);
				dictParserId = builder.addParser(dictParser);
			}
			Parser::ParserNetwork::ParserId arrayParserId;
			{
				Parser::RecursiveParser arrayParser(listParser);
				arrayParser.addEntryPoint("[", false);
				arrayParser.addExitPoint("]", false);
				arrayParserId = builder.addParser(arrayParser);
			}
			Parser::ParserNetwork::ParserId stringParserId;
			{
				Parser::RecursiveParser stringParser;
				for (EscapeMap::const_iterator it = escapeMap.begin(); it != escapeMap.end(); it++)
					stringParser.addEscapeSequence(it->second, std::string("") + it->first, Parser::RecursiveParser::ESCAPE_DIRECT_TO_TEXT);
				stringParser.addEntryPoint("\"", false);
				stringParser.addExitPoint("\"", false);
				stringParserId = builder.addParser(stringParser);
			}

			builder.insert(dictParserId, listParserId);
			builder.insert(arrayParserId, listParserId);
			builder.insert(stringParserId, listParserId);
			builder.insert(dictParserId, dictParserId);
			builder.insert(arrayParserId, dictParserId);
			builder.insert(stringParserId, dictParserId);
			builder.insert(dictParserId, arrayParserId);
			builder.insert(arrayParserId, arrayParserId);
			builder.insert(stringParserId, arrayParserId);

			builder.selectMainParser(listParserId);
		}

		static bool addToDict(const Object &item, std::ostream *errorLog, Dict *target, const std::string *key) {
			if (target->contains(*key)) {
				if (errorLog != NULL) (*errorLog) << "Dict has a duplicate entry for '" << (*key) << "'" << std::endl;
				return false;
			}
			target->set(*key, item);
			return true;
		}
		static bool addToArray(const Object &item, std::ostream *, Array *target) {
			target->push(item);
			return true;
		}

		template<typename Function, typename... Args>
		static bool parseToken(const Parser::RecursiveParser::Token &token, std::ostream *errorLog, Function function, Args... args) {
			bool failed = false;
			if (token.text == "true") {
				// Bool (true)
				Bool value;
				value.value() = true;
				if (!function(value, errorLog, args...)) failed = true; else return true;
			}
			else if (token.text == "false") {
				// Bool (false)
				Bool value;
				value.value() = false;
				if (!function(value, errorLog, args...)) failed = true; else return true;
			}
			else if (token.text == "null") {
				// Null
				Null value;
				if (!function(value, errorLog, args...)) failed = true; else return true;
			}
			else if (token.text != "") {
				// Probably a number or something...
				double numberValue;
				std::stringstream stream;
				stream << token.text << std::endl;
				stream >> numberValue;
				if (stream.fail()) {
					if (errorLog != NULL) (*errorLog) << "Could not categorize token '" << token.text << "'" << std::endl;
					failed = true;
				}
				else {
					Number value;
					value.value() = numberValue;
					if (!function(value, errorLog, args...)) failed = true; else return true;
				}
			}
			else if (token.entry == "{") {
				// Dict
				Dict value;
				if (token.subTokens.size() <= 0) { if (!function(value, errorLog, args...)) failed = true; else return true; }
				else if ((token.subTokens.size() % 4) != 3) {
					if (errorLog != NULL) (*errorLog) << "Dict has to be a certain amount of a:b couples, separated by ',' character...." << std::endl;
					failed = true;
				}
				else for (size_t i = 0; i < token.subTokens.size(); i += 4)
					if (!(token.subTokens[i].text == "" && token.subTokens[i].entry == "\"" && token.subTokens[i].subTokens.size() <= 1)) {
						if (errorLog != NULL) (*errorLog) << "Dict only accepts strings as keys" << std::endl;
						failed = true;
						break;
					}
				if (!failed) for (size_t i = 1; i < token.subTokens.size(); i += 4)
					if (token.subTokens[i].text != ":") {
						if (errorLog != NULL) (*errorLog) << "Expected ':' after a key" << std::endl;
						failed = true;
						break;
					}
				if (!failed) for (size_t i = 3; i < token.subTokens.size(); i += 4)
					if (token.subTokens[i].text != ",") {
						if (errorLog != NULL) (*errorLog) << "Expected ',' after a key : value pair" << std::endl;
						failed = true;
						break;
					}
				if (!failed) for (size_t i = 0; i < token.subTokens.size(); i += 4) {
					std::string key;
					if (token.subTokens[i].subTokens.size() <= 0) key = "";
					else key = token.subTokens[i].subTokens[0].text;
					if (!parseToken(token.subTokens[i + 2], errorLog, addToDict, &value, &key)) {
						failed = true;
						break;
					}
				}
				if (!failed) { if (!function(value, errorLog, args...)) failed = true; else return true; }
			}
			else if (token.entry == "[") {
				// Array
				Array value;
				if (token.subTokens.size() <= 0) { if (!function(value, errorLog, args...)) failed = true; else return true; }
				else if ((token.subTokens.size() % 2) != 1) {
					if (errorLog != NULL) (*errorLog) << "Array has to be a certain amount of entries, separated by ',' character...." << std::endl;
					failed = true;
				}
				else {
					for (size_t i = 1; i < token.subTokens.size(); i += 2)
						if (token.subTokens[i].text != ",") {
							if (errorLog != NULL) (*errorLog) << "Array elements should be separated by ',' character...." << std::endl;
							failed = true;
							break;
						}
					if (!failed) {
						for (size_t i = 0; i < token.subTokens.size(); i += 2)
							if (!parseToken(token.subTokens[i], errorLog, addToArray, &value)) {
								failed = true;
								break;
							}
						if (!failed) { if (!function(value, errorLog, args...)) failed = true; else return true; }
					}
				}
			}
			else if (token.entry == "\"") {
				// String
				if (token.subTokens.size() > 1) failed = true;
				else {
					String value;
					if (token.subTokens.size() <= 0) value.value() = "";
					else value.value() = token.subTokens[0].text;
					if (!function(value, errorLog, args...)) failed = true; else return true;
				}
			}
			else failed = true;
			if (!failed) return true;
			else {
				if (errorLog != NULL) {
					(*errorLog) << "Dson parser failed to decode token: " << std::endl;
					token.dump(*errorLog);
				}
				return false;
			}
		}
		static bool setObject(const Object &item, std::ostream *errorLog, Object **target) {
			if (item.type() == Object::DSON_DICT) (*target) = new Dict(*((Dict*)(&item)));
			else if (item.type() == Object::DSON_ARRAY) (*target) = new Array(*((Array*)(&item)));
			else if (item.type() == Object::DSON_STRING) (*target) = new String(*((String*)(&item)));
			else if (item.type() == Object::DSON_NUMBER) (*target) = new Number(*((Number*)(&item)));
			else if (item.type() == Object::DSON_BOOL) (*target) = new Bool(*((Bool*)(&item)));
			else if (item.type() == Object::DSON_NULL) (*target) = new Null(*((Null*)(&item)));
			else {
				if (errorLog != NULL) (*errorLog) << "Item type not recognized: " << item.toString() << std::endl;
				return false;
			}
			return true;
		}
	}
	
	Object* parse(const std::string &text, std::ostream *errorLog) {
		std::vector<Parser::RecursiveParser::Token> tokens = Parser::ParserNetwork(buildDsonParserNetwork).parse(text);
		if (tokens.size() < 1) {
			if (errorLog != NULL) (*errorLog) << "No in tokens (source: \"" << text << "\")" << std::endl;
			return NULL;
		}
		if (tokens.size() > 1) {
			if (errorLog != NULL) (*errorLog) << "Multiple entries in tokens (source: \"" << text << "\")" << std::endl;
			return NULL;
		}
		else {
			Object *result;
			if (parseToken(tokens[0], errorLog, setObject, &result)) return result;
			else return NULL;
		}
	}
}

