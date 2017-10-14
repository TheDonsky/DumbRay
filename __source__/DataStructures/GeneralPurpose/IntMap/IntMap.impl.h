#include"IntMap.h"



template<typename Type, unsigned int base>
__device__ __host__ inline IntMap<Type, base>::IntMap(){
	clear();
}


template<typename Type, unsigned int base>
__device__ __host__ inline bool IntMap<Type, base>::contains(int key)const{
	return(find(key) >= 0);
}
template<typename Type, unsigned int base>
__device__ __host__ inline Type& IntMap<Type, base>::operator[](int key){
	return nodes[forceFind(key)].value;
}
template<typename Type, unsigned int base>
__device__ __host__ inline const Type& IntMap<Type, base>::operator[](int key)const{
	return nodes[find(key)].value;
}
template<typename Type, unsigned int base>
__device__ __host__ inline void IntMap<Type, base>::put(int key, Type value){
	nodes[forceFind(key)].value = value;
}
template<typename Type, unsigned int base>
__device__ __host__ inline void IntMap<Type, base>::remove(int key){
	int index = find(key);
	if (index >= 0) nodes[index].containsValue = false;
}
template<typename Type, unsigned int base>
__device__ __host__ inline void IntMap<Type, base>::clear() {
	nodes.clear(); 
	nodes.push(Node());
	nodes.push(Node());
}





template<typename Type, unsigned int base>
__device__ __host__ inline int IntMap<Type, base>::forceFind(int key){
	int index = getStart(key);
	while (true){
		if (key == 0){
			nodes[index].containsValue = true;
			return index;
		}
		else if (nodes[index].children <= 0){
			nodes[index].children = nodes.size();
			nodes.flush(base);
		}
		index = nodes[index].children + key % base;
		key /= base;
	}
}
template<typename Type, unsigned int base>
__device__ __host__ inline int IntMap<Type, base>::find(int key) const{
	int index = getStart(key);
	while (true){
		if (key == 0){
			if (nodes[index].containsValue) return index;
			else return -1;
		}
		else if (nodes[index].children <= 0) return -1;
		index = nodes[index].children + key % base;
		key /= base;
	}
}
template<typename Type, unsigned int base>
__device__ __host__ inline int IntMap<Type, base>::getStart(int &key) const{
	if (key < 0){
		key = -key;
		return 1;
	}
	return 0;
}



template<typename Type, unsigned int base>
__device__ __host__ inline IntMap<Type, base>::Node::Node(){
	containsValue = false;
	children = 0;
}


