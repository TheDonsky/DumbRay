#include "Dson.h"
#include<sstream>
#include<new>



namespace Dson {
	namespace {
		typedef std::unordered_map<char, std::string> EscapeMap;
		static EscapeMap getEscapeMap() {
			std::unordered_map<char, std::string> result;
			result['"'] = "\\\"";
			result['\\'] = "\\\\";
			result['\/'] = "\\\/";
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
		if (value.type() == DUMBSON_DICT) map[key] = new Dict(*((Dict*)(&value)));
		else if (value.type() == DUMBSON_ARRAY) map[key] = new Array(*((Array*)(&value)));
		else if (value.type() == DUMBSON_STRING) map[key] = new String(*((String*)(&value)));
		else if (value.type() == DUMBSON_NUMBER) map[key] = new Number(*((Number*)(&value)));
		else if (value.type() == DUMBSON_BOOL) map[key] = new Bool(*((Bool*)(&value)));
		else if (value.type() == DUMBSON_NULL) map[key] = new Null(*((Null*)(&value)));
	}
	void Dict::remove(const std::string &key) {
		std::unordered_map<std::string, Object*>::const_iterator it = map.find(key);
		if (it != map.end()) {
			delete it->second;
			map.erase(it);
		}
	}
	Dict::Type Dict::type()const { return DUMBSON_DICT; }
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
		if (value.type() == DUMBSON_DICT) objects[index] = new Dict(*((Dict*)(&value)));
		else if (value.type() == DUMBSON_ARRAY) objects[index] = new Array(*((Array*)(&value)));
		else if (value.type() == DUMBSON_STRING) objects[index] = new String(*((String*)(&value)));
		else if (value.type() == DUMBSON_NUMBER) objects[index] = new Number(*((Number*)(&value)));
		else if (value.type() == DUMBSON_BOOL) objects[index] = new Bool(*((Bool*)(&value)));
		else if (value.type() == DUMBSON_NULL) objects[index] = new Null(*((Null*)(&value)));
	}
	void Array::push(const Object &value) { set(size(), value); }
	void Array::pop() {
		if (objects[size() - 1] != NULL) delete objects[size() - 1];
		objects.pop_back();
	}
	Array::Type Array::type()const { return DUMBSON_ARRAY; }
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
	String::Type String::type()const { return DUMBSON_STRING; }
	std::string String::toString(const std::string &, const std::string &)const {
		return stringToJsonSourceString(val);
	}
	




	int Number::intValue()const { return ((int)val); }
	float Number::floatValue()const { return ((float)val); }
	double& Number::value() { return val; }
	double Number::value()const { return val; }
	Number::Type Number::type()const { return DUMBSON_NUMBER; }
	std::string Number::toString(const std::string &, const std::string &)const {
		std::stringstream stream;
		stream << val;
		return stream.str();
	}





	bool &Bool::value() { return val; }
	bool Bool::value()const { return val; }
	Bool::Type Bool::type()const { return DUMBSON_BOOL; }
	std::string Bool::toString(const std::string &, const std::string &)const { return (val ? "true" : "false"); }
	




	Null::Type Null::type()const { return DUMBSON_NULL; }
	std::string Null::toString(const std::string &, const std::string &)const { return "null"; }





	namespace {

	}
	
	Object* parse(const std::string &text) {
		return NULL;
	}
}

