#include "ByteSet.h"


namespace Parser {

	ByteSet::ByteSet() {
		clean();
	}
	void ByteSet::clean() {
		nodes.clear();
		nodes.push_back(Node());
	}


	ByteSet::NodeId ByteSet::addString(const char *string, bool inverse, int maxSize) {
		char zero = '\0';
		return addKey(string, maxSize, sizeof(char), !inverse, &zero);
	}

	ByteSet::NodeId ByteSet::findString(const char *string, bool inverse, int maxSize, bool firstPrefix) const {
		char zero = '\0';
		return findKey(string, maxSize, sizeof(char), !inverse, &zero, firstPrefix);
	}

	void ByteSet::removeString(const char *string, bool inverse, int maxSize, bool firstPrefix) {
		char zero = '\0';
		removeKey(string, maxSize, sizeof(char), !inverse, &zero, firstPrefix);
	}

	void ByteSet::removeId(NodeId id) {
		if (!BYTE_SET_NOT_A_NODE(id))
			nodes[id].used = false;
	}


	ByteSet::Flags *ByteSet::stringFlags(const char *string, bool inverse, int maxSize, bool firstPrefix) {
		char zero = '\0';
		return findFlags(string, maxSize, sizeof(char), !inverse, &zero, firstPrefix);
	}
	const ByteSet::Flags *ByteSet::stringFlags(const char *string, bool inverse, int maxSize, bool firstPrefix)const {
		char zero = '\0';
		return findFlags(string, maxSize, sizeof(char), !inverse, &zero, firstPrefix);
	}


	ByteSet::Flags &ByteSet::flags(NodeId id) {
		return nodes[id].flags;
	}
	const ByteSet::Flags &ByteSet::flags(NodeId id) const {
		return nodes[id].flags;
	}





	ByteSet::Node::Node() {
		clean();
	}
	void ByteSet::Node::clean() {
		for (unsigned int i = 0; i < 16; i++)
			children[i] = -1;
		flags = 0;
		used = false;
	}



	static bool isSentinel(const char *current, const char *sentinel, unsigned int elemSize) {
		bool sentinelMet = (sentinel != NULL);
		if (sentinelMet)
			for (unsigned int i = 0; i < elemSize; i++)
				if (current[i] != sentinel[i]) {
					sentinelMet = false;
					break;
				}
		return sentinelMet;
	}

#define SEARCH_OR_INSERT_INITIALIZE_VARS \
	NodeId node = 0; \
	int delta = ((ascending ? 1 : (-1)) * elemSize); \
	const char *end = ((size >= 0) ? (key + (size * delta)) : NULL)
#define SEARCH_OR_INSERT_MAIN_LOOP \
	const char *current = key; current != end; current += delta
#define IS_SENTINEL isSentinel(current, sentinel, elemSize)

	ByteSet::NodeId ByteSet::addKey(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel) {
		SEARCH_OR_INSERT_INITIALIZE_VARS;
		for (SEARCH_OR_INSERT_MAIN_LOOP) {
			if (IS_SENTINEL) break;
			for (unsigned int i = 0; i < elemSize; i++) {
				unsigned char byte = (((unsigned char *)current)[i]);
				for (int j = 0; j < 2; j++) {
					unsigned char bits = (byte % 16);
					byte /= 16;
					NodeId index = nodes[node].children[bits];
					if (index < 0) {
						index = (NodeId)nodes.size();
						nodes.push_back(Node());
						nodes[node].children[bits] = index;
					}
					node = index;
				}
			}
		}
		nodes[node].used = true;
		return node;
	}

	ByteSet::NodeId ByteSet::findKey(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel, bool firstPrefix) const {
		SEARCH_OR_INSERT_INITIALIZE_VARS;
		for (SEARCH_OR_INSERT_MAIN_LOOP) {
			if (IS_SENTINEL) {
				if (nodes[node].used) return node;
				else return BYTE_SET_NO_NODE;
			}
			if (firstPrefix && nodes[node].used)
				return node;
			for (unsigned int i = 0; i < elemSize; i++) {
				unsigned char byte = (((unsigned char *)current)[i]);
				for (int j = 0; j < 2; j++) {
					unsigned char bits = (byte % 16);
					byte /= 16;
					node = nodes[node].children[bits];
					if (BYTE_SET_NOT_A_NODE(node))
						return BYTE_SET_NO_NODE;
				}
			}
		}
		return (nodes[node].used ? node : BYTE_SET_NO_NODE);
	}

#define FIND_KEY NodeId node = findKey(key, size, elemSize, ascending, sentinel, firstPrefix)

	void ByteSet::removeKey(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel, bool firstPrefix) {
		FIND_KEY;
		if (!BYTE_SET_NOT_A_NODE(node))
			nodes[node].used = false;
	}

#define FIND_FLAGS_LOGIC \
	FIND_KEY; \
	return (BYTE_SET_NOT_A_NODE(node) ? NULL : (&(nodes[node].flags)))
	ByteSet::Flags *ByteSet::findFlags(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel, bool firstPrefix) {
		FIND_FLAGS_LOGIC;
	}
	const ByteSet::Flags *ByteSet::findFlags(const char *key, int size, unsigned int elemSize, bool ascending, const char *sentinel, bool firstPrefix) const {
		FIND_FLAGS_LOGIC;
	}

}
