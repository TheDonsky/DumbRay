#pragma once
#include<vector>
#include <cstddef>


namespace Parser {

#define BYTE_SET_NO_NODE -1
#define BYTE_SET_NOT_A_NODE(node) (node < 0)

	class ByteSet {
	public:
		typedef int NodeId;
		typedef int Flags;

	public:
		ByteSet();
		void clean();

		template<typename Type>
		inline NodeId add(const Type *startAddress, int count, bool orderAscending = true, const Type *termialValue = NULL);
		template<typename Type>
		inline NodeId add(const Type &value);
		NodeId addString(const char *string, bool inverse = false, int maxSize = -1);

		template<typename Type>
		inline NodeId find(const Type *startAddress, int count, bool orderAscending = true, const Type *termialValue = NULL, bool firstPrefix = false)const;
		template<typename Type>
		inline NodeId find(const Type &value)const;
		NodeId findString(const char *string, bool inverse = false, int maxSize = -1, bool firstPrefix = false)const;

		template<typename Type>
		inline void remove(const Type *startAddress, int count, bool orderAscending = true, const Type *termialValue = NULL, bool firstPrefix = false);
		template<typename Type>
		inline void remove(const Type &value);
		void removeString(const char *string, bool inverse = false, int maxSize = -1, bool firstPrefix = false);
		void removeId(NodeId id);

		template<typename Type>
		inline Flags *flagsOf(const Type *startAddress, int count, bool orderAscending = true, const Type *termialValue = NULL, bool firstPrefix = false);
		template<typename Type>
		const inline Flags *flagsOf(const Type *startAddress, int count, bool orderAscending = true, const Type *termialValue = NULL, bool firstPrefix = false)const;
		template<typename Type>
		inline Flags *flagsOf(const Type &value);
		template<typename Type>
		const inline Flags *flagsOf(const Type &value)const;
		Flags *stringFlags(const char *string, bool inverse = false, int maxSize = -1, bool firstPrefix = false);
		const Flags *stringFlags(const char *string, bool inverse = false, int maxSize = -1, bool firstPrefix = false)const;

		Flags &flags(NodeId id);
		const Flags &flags(NodeId id)const;

	private:
		struct Node {
			NodeId children[16];
			Flags flags;
			bool used;

			Node();
			void clean();
		};
		std::vector<Node> nodes;

		NodeId addKey(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel);
		NodeId findKey(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel, bool firstPrefix)const;
		void removeKey(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel, bool firstPrefix);
		Flags *findFlags(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel, bool firstPrefix);
		const Flags *findFlags(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel, bool firstPrefix)const;
	};








	// DON'T BOTHER READING BELOW....

	template<typename Type>
	inline ByteSet::NodeId ByteSet::add(const Type *startAddress, int count, bool orderAscending, const Type *termialValue) {
		return addKey((const char*)startAddress, count, sizeof(Type), orderAscending, (const char*)termialValue);
	}
	template<typename Type>
	inline ByteSet::NodeId ByteSet::add(const Type &value) {
		return addKey((const char*)(&value), 1, sizeof(Type), true, NULL);
	}
	template<typename Type>
	inline ByteSet::NodeId ByteSet::find(const Type *startAddress, int count, bool orderAscending, const Type *termialValue, bool firstPrefix) const {
		return findKey((const char*)startAddress, count, sizeof(Type), orderAscending, (const char*)termialValue, firstPrefix);
	}
	template<typename Type>
	inline ByteSet::NodeId ByteSet::find(const Type &value) const {
		return findKey((const char*)(&value), 1, sizeof(Type), true, NULL, false);
	}
	template<typename Type>
	inline void ByteSet::remove(const Type *startAddress, int count, bool orderAscending, const Type *termialValue, bool firstPrefix) {
		removeKey((const char*)startAddress, count, sizeof(Type), orderAscending, (const char*)termialValue, firstPrefix);
	}
	template<typename Type>
	inline void ByteSet::remove(const Type &value) {
		removeKey((const char*)(&value), 1, sizeof(Type), true, NULL, false);
	}
	template<typename Type>
	inline ByteSet::Flags *ByteSet::flagsOf(const Type *startAddress, int count, bool orderAscending, const Type *termialValue, bool firstPrefix) {
		return findFlags((const char*)startAddress, count, sizeof(Type), orderAscending, (const char*)termialValue, firstPrefix);
	}
	template<typename Type>
	const inline ByteSet::Flags *ByteSet::flagsOf(const Type *startAddress, int count, bool orderAscending, const Type *termialValue, bool firstPrefix)const {
		return findFlags((const char*)startAddress, count, sizeof(Type), orderAscending, (const char*)termialValue, firstPrefix);
	}
	template<typename Type>
	inline ByteSet::Flags *ByteSet::flagsOf(const Type &value) {
		return findFlags((const char*)(&value), 1, sizeof(Type), true, NULL, false);
	}
	template<typename Type>
	const inline ByteSet::Flags *ByteSet::flagsOf(const Type &value)const {
		return findFlags((const char*)(&value), 1, sizeof(Type), true, NULL, false);
	}

}
