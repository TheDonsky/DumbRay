#include"Vector.h"



template<typename Type>
inline Vector<Type>::Vector() {
	data = NULL;
	alloc = 0;
	used = 0;
}
template<typename Type>
inline Vector<Type>::Vector(const Vector &v) : Vector(){
	(*this) = v;
}
template<typename Type>
inline Vector<Type>& Vector<Type>::operator=(const Vector &v) {
	if (demandMinCapacity(v.alloc)) {
		for (size_t i = 0; i < v.used; i++)
			data[i] = v.data[i];
		used = v.used;
	}
	return (*this);
}
template<typename Type>
inline Vector<Type>::Vector(Vector &&v) : Vector(){
	(*this) = v;
}
template<typename Type>
inline Vector<Type>& Vector<Type>::operator=(Vector &&v) {
	Type *tmp_t = data; data = v.data; v.data = tmp_t;
	size_t tmp_s = alloc; alloc = v.alloc; v.alloc = tmp_s;
	tmp_s = used; used = v.used; v.used = tmp_s;
	return (*this);
}
template<typename Type>
inline Vector<Type>::~Vector() {
	if (data != NULL) delete[] data;
}

template<typename Type>
inline bool Vector<Type>::push(const Type &elem) {
	if (demandMinCapacity(used + 1)) {
		data[used] = elem;
		used++;
		return true;
	}
	else return false;
}
template<typename Type>
inline Type& Vector<Type>::pop() {
	used--;
	return data[used];
}
template<typename Type>
inline Type& Vector<Type>::top() {
	return data[used - 1];
}
template<typename Type>
inline const Type& Vector<Type>::top()const {
	return data[used - 1];
}
template<typename Type>
inline void Vector<Type>::clear() {
	used = 0;
}

template<typename Type>
inline size_t Vector<Type>::size()const {
	return used;
}
template<typename Type>
inline Type& Vector<Type>::operator[](size_t index) {
	return data[index];
}
template<typename Type>
inline const Type& Vector<Type>::operator[](size_t index)const {
	return data[index];
}
template<typename Type>
inline Type* Vector<Type>::operator+(size_t offset) {
	return (data + offset);
}
template<typename Type>
inline const Type* Vector<Type>::operator+(size_t offset)const {
	return (data + offset);
}





template<typename Type>
inline bool Vector<Type>::demandMinCapacity(size_t capacity) {
	size_t newCapacity = alloc;
	if (newCapacity <= 0) newCapacity = 2;
	while (newCapacity < capacity) newCapacity *= 2;
	Type *newData = new Type[newCapacity];
	if (newData != NULL) {
		for (int i = 0; i < used; i++)
			newData[i] = data[i];
		delete[] data;
		data = newData;
		alloc = newCapacity;
	}
	else return false;
}
