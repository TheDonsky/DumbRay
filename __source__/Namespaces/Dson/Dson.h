#pragma once
#include<string>
#include<vector>
#include<unordered_map>
#include <iostream>


namespace Dson {
	class Object {
	public:
		enum Type {
			DSON_DICT,
			DSON_ARRAY,
			DSON_STRING,
			DSON_NUMBER,
			DSON_BOOL,
			DSON_NULL
		};

	public:
		virtual ~Object();

		virtual Type type()const = 0;
		virtual std::string toString(const std::string &baseInset = "\t", const std::string &inset = "")const = 0;

		template<typename DsonType>
		inline DsonType *safeConvert(std::ostream *errorStream = NULL, const std::string &error = "Dson class mismatch...", Type targetType = DsonType().type()) {
			if (type() == targetType) return ((DsonType*)this);
			else {
				if (errorStream != NULL) (*errorStream) << error << std::endl;
				return NULL;
			}
		}
		template<typename DsonType>
		inline const DsonType *safeConvert(std::ostream *errorStream = NULL, const std::string &error = "Dson class mismatch...", Type targetType = DsonType().type()) const {
			if (type() == targetType) return ((const DsonType*)this);
			else {
				if (errorStream != NULL) (*errorStream) << error << std::endl;
				return NULL;
			}
		}
	private:
	};

	class Dict : public Object {
	public:
		Dict();
		Dict(const Dict &other);
		Dict &operator=(const Dict &other);
		void copyFrom(const Dict &other);
		Dict(Dict &&other);
		Dict &operator=(Dict &&other);
		void stealFrom(Dict &other);
		virtual ~Dict();

		size_t size()const;
		bool contains(const std::string &key)const;
		const Object& get(const std::string &key)const;
		const Object& operator[](const std::string &key)const;

		void set(const std::string &key, const Object &value);
		void remove(const std::string &key);

		virtual Type type()const;
		virtual std::string toString(const std::string &baseInset = "\t", const std::string &inset = "")const;


	private:
		std::unordered_map<std::string, Object*> map;
	};

	class Array : public Object {
	public:
		Array();
		Array(const Array &other);
		Array &operator=(const Array &other);
		void copyFrom(const Array &other);
		Array(Array &&other);
		Array &operator=(Array &&other);
		void stealFrom(Array &other);
		virtual ~Array();

		size_t size()const;
		const Object& get(size_t index)const;
		const Object& operator[](size_t index)const;

		void set(size_t index, const Object &value);
		void push(const Object &value);
		void pop();

		virtual Type type()const;
		virtual std::string toString(const std::string &baseInset = "\t", const std::string &inset = "")const;
	private:
		std::vector<Object*> objects;
	};

	class String : public Object {
	public:
		std::string &value();
		const std::string &value()const;

		virtual Type type()const;
		virtual std::string toString(const std::string &baseInset = "\t", const std::string &inset = "")const;
	private:
		std::string val;
	};

	class Number : public Object {
	public:
		int intValue()const;
		float floatValue()const;
		double& value();
		double value()const;

		virtual Type type()const;
		virtual std::string toString(const std::string &baseInset = "\t", const std::string &inset = "")const;
	private:
		double val;
	};

	class Bool : public Object {
	public:
		bool &value();
		bool value()const;

		virtual Type type()const;
		virtual std::string toString(const std::string &baseInset = "\t", const std::string &inset = "")const;
	private:
		bool val;
	};

	class Null : public Object {
	public:
		virtual Type type()const;
		virtual std::string toString(const std::string &baseInset = "\t", const std::string &inset = "")const;
	private:
	};

	Object* parse(const std::string &text, std::ostream *errorLog = NULL);
}
