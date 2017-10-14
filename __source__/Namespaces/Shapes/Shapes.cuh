#pragma once

#include"../../DataStructures/Primitives/Pure/Vector3/Vector3.h"
#include"../../DataStructures/Primitives/Compound/AABB/AABB.h"
#include"../../DataStructures/Primitives/Compound/Triangle/Triangle.h"
#include"../../DataStructures/Objects/Meshes/BakedTriMesh/BakedTriMesh.h"
#include"../../DataStructures/Primitives/Compound/Ray/Ray.h"
#include"../../DataStructures/GeneralPurpose/Stacktor/Stacktor.cuh"




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
	//*
	template<typename Type, typename BoundType>
	__dumb__ bool sharePoint(const Type &a, const Type &b, const BoundType &commonPointBounds);
	//*/
	template<typename Type1, typename Type2>
	// Returns mass center of intersection, if it exists, or some undefined value.
	__dumb__ Vertex intersectionCenter(const Type1 &a, const Type2 &b);
	template<typename Type1, typename Type2>
	// Returns bounding box of intersection, if it exists, or some undefined value.
	__dumb__ AABB intersectionBounds(const Type1 &a, const Type2 &b);

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
