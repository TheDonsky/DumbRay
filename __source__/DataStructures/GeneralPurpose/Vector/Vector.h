#pragma once



template<typename Type>
class Vector {
public:
	inline Vector();
	inline Vector(const Vector &v);
	inline Vector& operator=(const Vector &v);
	inline Vector(Vector &&v);
	inline Vector& operator=(Vector &&v);
	inline ~Vector();

	inline bool push(const Type &elem);
	inline Type& pop();
	inline Type& top();
	inline const Type& top()const;
	inline void clear();

	inline size_t size()const;
	inline Type& operator[](size_t index);
	inline const Type& operator[](size_t index)const;
	inline Type* operator+(size_t offset);
	inline const Type* operator+(size_t offset)const;

private:
	Type *data;
	size_t alloc, used;

	inline bool demandMinCapacity(size_t capacity);
};




#include"Vector.impl.h"
