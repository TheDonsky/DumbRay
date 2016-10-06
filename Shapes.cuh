#pragma once

#include"Vector3.h"
#include"AABB.h"
#include"Triangle.h"
#include"BakedTriMesh.h"
#include"Ray.h"




namespace Shapes{

	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	template<typename Type1, typename Type2>
	__dumb__ bool intersect(const Type1 &first, const Type2 &second);


	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	template<typename Type>
	__dumb__ bool cast(const Ray &ray, const Type &object, bool clipBackface);

	template<typename Type>
	__dumb__ bool castPreInversed(const Ray &inversedRay, const Type &object, bool clipBackface);

	template<typename Type>
	__dumb__ bool cast(const Ray &ray, const Type &object, float &hitDistance, Vertex &hitPoint, bool clipBackface);


	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	template<typename Type, typename BoundType>
	__dumb__ bool sharePoint(const Type &a, const Type &b, const BoundType &commonPointBounds);


	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	template<typename Type>
	__dumb__ Vertex massCenter(const Type &shape);
	template<typename Type>
	__dumb__ AABB boundingBox(const Type &shape);


	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	template<typename Type>
	__dumb__ void dump(const Type &shape);
}





#include"Shapes.impl.cuh"
