#pragma once
#include "ByteSet.h"

namespace Parser {

	template<typename Type>
	class ByteMap {
	public:
		typedef ByteSet::NodeId NodeId;


	public:
		inline void clean();

		template<typename KeyType>
		inline NodeId add(const Type &value, const KeyType *startAddress, int count, bool orderAscending = true, const KeyType *termialValue = NULL);
		template<typename KeyType>
		inline NodeId add(const Type &value, const KeyType &key);
		inline NodeId addString(const Type &value, const char *string, bool inverse = false, int maxSize = -1);

		template<typename KeyType>
		inline NodeId find(const KeyType *startAddress, int count, bool orderAscending = true, const KeyType *termialValue = NULL, bool firstPrefix = false)const;
		template<typename KeyType>
		inline NodeId find(const KeyType &key)const;
		inline NodeId findString(const char *string, bool inverse = false, int maxSize = -1, bool firstPrefix = false)const;

		template<typename KeyType>
		inline void remove(const KeyType *startAddress, int count, bool orderAscending = true, const KeyType *termialValue = NULL, bool firstPrefix = false);
		template<typename KeyType>
		inline void remove(const KeyType &key);
		inline void removeString(const char *string, bool inverse = false, int maxSize = -1, bool firstPrefix = false);

		template<typename KeyType>
		inline Type *valueOf(const KeyType *startAddress, int count, bool orderAscending = true, const KeyType *termialValue = NULL, bool firstPrefix = false);
		template<typename KeyType>
		const inline Type *valueOf(const KeyType *startAddress, int count, bool orderAscending = true, const KeyType *termialValue = NULL, bool firstPrefix = false)const;
		template<typename KeyType>
		inline Type *valueOf(const KeyType &key);
		template<typename KeyType>
		const inline Type *valueOf(const KeyType &key)const;
		Type *stringValue(const char *string, bool inverse = false, int maxSize = -1, bool firstPrefix = false);
		const Type *stringValue(const char *string, bool inverse = false, int maxSize = -1, bool firstPrefix = false)const;

		Type &value(NodeId id);
		const Type &value(NodeId id)const;

	private:
		struct OwnedData {
			NodeId owner;
			Type data;
			inline OwnedData(const NodeId &o, const Type &d);
		};
		ByteSet set;
		std::vector<OwnedData> data;
	};

}





#include"ByteMap.impl.h"
