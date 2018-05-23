#include "ByteMap.h"


namespace Parser {

	template<typename Type>
	inline void ByteMap<Type>::clean() {
		set.clean();
		data.clear();
	}

	template<typename Type>
	template<typename KeyType>
	inline typename ByteMap<Type>::NodeId ByteMap<Type>::add(
		const Type &value,
		const KeyType *startAddress,
		int count,
		bool orderAscending,
		const KeyType *termialValue) {
		NodeId id = set.add<KeyType>(startAddress, count, orderAscending, termialValue);
		set.flags(id) = (ByteSet::Flags)data.size();
		data.push_back(OwnedData(id, value));
		return id;
	}
	template<typename Type>
	template<typename KeyType>
	inline typename ByteMap<Type>::NodeId ByteMap<Type>::add(const Type &value, const KeyType &key) {
		return add<KeyType>(value, &key, 1, true, NULL);
	}
	template<typename Type>
	inline typename ByteMap<Type>::NodeId ByteMap<Type>::addString(const Type &value, const char *string, bool inverse, int maxSize) {
		const char zero = '\0';
		return add<char>(value, string, maxSize, !inverse, &zero);
	}

	template<typename Type>
	template<typename KeyType>
	inline typename ByteMap<Type>::NodeId ByteMap<Type>::find(
		const KeyType *startAddress,
		int count,
		bool orderAscending,
		const KeyType *termialValue,
		bool firstPrefix)const {
		return set.find<KeyType>(startAddress, count, orderAscending, termialValue, firstPrefix);
	}
	template<typename Type>
	template<typename KeyType>
	inline typename ByteMap<Type>::NodeId ByteMap<Type>::find(const KeyType &key)const {
		return find<KeyType>(&key, 1, true, NULL, false);
	}
	template<typename Type>
	inline typename ByteMap<Type>::NodeId ByteMap<Type>::findString(const char *string, bool inverse, int maxSize, bool firstPrefix)const {
		const char zero = '\0';
		return find<char>(string, maxSize, !inverse, &zero, firstPrefix);
	}

	template<typename Type>
	template<typename KeyType>
	inline void ByteMap<Type>::remove(
		const KeyType *startAddress,
		int count,
		bool orderAscending,
		const KeyType *termialValue,
		bool firstPrefix) {
		NodeId id = set.find<KeyType>(startAddress, count, orderAscending, termialValue, firstPrefix);
		if (BYTE_SET_NOT_A_NODE(id)) return;
		int dataId = set.flags(id);
		OwnedData &d = data[dataId];
		d = data.back();
		set.flags(d.owner) = dataId;
		data.pop_back();
		set.removeId(id);
	}
	template<typename Type>
	template<typename KeyType>
	inline void ByteMap<Type>::remove(const KeyType &key) {
		remove<KeyType>(&key, 1, true, NULL, false);
	}
	template<typename Type>
	inline void ByteMap<Type>::removeString(const char *string, bool inverse, int maxSize, bool firstPrefix) {
		const char zero = '\0';
		remove<char>(string, maxSize, !inverse, &zero, firstPrefix);
	}

	template<typename Type>
	template<typename KeyType>
	inline Type *ByteMap<Type>::valueOf(
		const KeyType *startAddress,
		int count,
		bool orderAscending,
		const KeyType *termialValue,
		bool firstPrefix) {
		NodeId nodeId = set.find<KeyType>(startAddress, count, orderAscending, termialValue, firstPrefix);
		return (BYTE_SET_NOT_A_NODE(nodeId) ? NULL : (&value(nodeId)));
	}
	template<typename Type>
	template<typename KeyType>
	const inline Type *ByteMap<Type>::valueOf(
		const KeyType *startAddress,
		int count,
		bool orderAscending,
		const KeyType *termialValue,
		bool firstPrefix)const {
		return ((const Type*)(((ByteMap<Type>*)this)->valueOf<KeyType>(startAddress, count, orderAscending, termialValue, firstPrefix)));
	}
	template<typename Type>
	template<typename KeyType>
	inline Type *ByteMap<Type>::valueOf(const KeyType &key) {
		return valueOf<KeyType>(&key, 1, true, NULL, false);
	}
	template<typename Type>
	template<typename KeyType>
	const inline Type *ByteMap<Type>::valueOf(const KeyType &key)const {
		return valueOf<KeyType>(&key, 1, true, NULL, false);
	}
	template<typename Type>
	Type *ByteMap<Type>::stringValue(const char *string, bool inverse, int maxSize, bool firstPrefix) {
		const char zero = '\0';
		return valueOf<char>(string, maxSize, !inverse, &zero, firstPrefix);
	}
	template<typename Type>
	const Type *ByteMap<Type>::stringValue(const char *string, bool inverse, int maxSize, bool firstPrefix)const {
		const char zero = '\0';
		return valueOf<char>(string, maxSize, !inverse, &zero, firstPrefix);
	}

	template<typename Type>
	Type &ByteMap<Type>::value(NodeId id) {
		return data[set.flags(id)].data;
	}
	template<typename Type>
	const Type &ByteMap<Type>::value(NodeId id)const {
		return data[set.flags(id)].data;
	}


	template<typename Type>
	inline ByteMap<Type>::OwnedData::OwnedData(const NodeId &o, const Type &d) {
		owner = o;
		data = d;
	}

}
