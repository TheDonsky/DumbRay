#pragma once
#include"../Ray/Ray.h"
#include"../Triangle/Triangle.h"





/*
Axis aligned bounding box
PS: can be used as a typename "AxisAlignedBoundingBox" as well.
*/
class AABB{
public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Construction: **/
	// Default constructor (does nothing)
	__dumb__ AABB();
	// Constructor (minimal and maximal boundaries will be defined by s and e)
	__dumb__ AABB(const Vertex &s, const Vertex &e);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Getters and setters: **/


	/** ========================================================== **/
	/*| Getters |*/

	// Returns the center
	__dumb__ Vertex getCenter()const;
	// Returns the minimum boundary
	__dumb__ Vertex getMin()const;
	// Returns the maximum boundary
	__dumb__ Vertex getMax()const;
	// Returns the extents (dimensions against center(half of the total size))
	__dumb__ Vector3 getExtents()const;
	// Returns the dimensions of the bounding box
	__dumb__ Vector3 getSize()const;


	/** ========================================================== **/
	/*| Setters |*/

	// Sets the center (moves both min and max, while reserving size)
	__dumb__ void setCenter(const Vertex &v);
	// Sets the minimum boundary (maximum boundary is reserved, or swapped, if the resulting AABB runs a risc of having negative dimensions)
	__dumb__ void setMin(const Vertex &v);
	// Sets the maximum boundary (minimum boundary is reserved, or swapped, if the resulting AABB runs a risc of having negative dimensions)
	__dumb__ void setMax(const Vertex &v);
	// Sets the extents (dimensions against center(half of the total size); center is reserved)
	__dumb__ void setExtents(const Vector3 &v);
	// Sets the dimensions of the bounding box (center is reserved)
	__dumb__ void setSize(const Vector3 &v);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Intersections: **/
	// Checks, if the axis aligned bounding box contains the Vertex
	__dumb__ bool contains(const Vertex &v)const;
	// Checks, if the axis aligned bounding box intersects the given triangle
	__dumb__ bool intersects(const Triangle &t)const;
	template <typename Type>
	// Generic intersection check
	__dumb__ bool intersects(const Type &t)const;
	// Checks, if the axis aligned bounding box intersects the given triangle, defined by vertexes
	__dumb__ bool intersectsTriangle(const Vertex &a, const Vertex &b, const Vertex &c)const;
	// Returns the distance the ray needs to travel in order to hit the box (0 or negative if inside, FLT_MAX, if hit can't occure)
	__dumb__ float cast(const Ray &r)const;
	// Returns the distance the ray needs to travel in order to hit the box (ray should have pre-inversed direction vector(gives some speedup); 0 or negative if inside, FLT_MAX, if hit can't occure)
	__dumb__ float castPreInversed(const Ray &r)const;





private:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Parameters: **/
	Vertex start, end;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Private functions: **/
	// checks, if start.x <= end.x, start.y <= end.y and start.z < end.z and swaps values if any of these requirements is not met
	__dumb__ void fix();
	template<unsigned int dimm>
	// checks the triangle intersection (if dimm is 0, for all 3 dimensions, if its 1, for x and y only and, in case of 2, just y)
	__dumb__ bool intersectsTri(Triangle t)const;
	template<unsigned int dimm>
	__dumb__ bool intersectsTri(const Triangle &t, float av, float bv, float cv, float s, float e)const;
};





typedef AABB AxisAlignedBoundingBox;





#include"AABB.impl.h"
