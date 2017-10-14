#pragma once
#include<math.h>
#include"../../Pure/Vector3/Vector3.h"
#include"../../../Objects/Components/Transform/Transform.h"
#include"../Ray/Ray.h"





struct Triangle{
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Parameters: **/
	Vertex a, b, c;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Construction: **/
	// Does nothing
	__dumb__ Triangle();
	// Simple constructor
	__dumb__ Triangle(const Vertex &A, const Vertex &B, const Vertex &C);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Operators: **/


	/** ========================================================== **/
	/*| + |*/
	// Same as t
	__dumb__ Triangle operator+()const;
	// Sum (Triangle(a + t.a, b + t.b, c +_ t.c))
	__dumb__ Triangle operator+(const Triangle &t)const;
	// Shifts triangle node by node
	__dumb__ Triangle& operator+=(const Triangle &t);
	// Same Triangle, shifted by v
	__dumb__ Triangle operator+(const Vector3 &v)const;
	// Shifts Triangle by v
	__dumb__ Triangle& operator+=(const Vector3 &v);
	
	/** ========================================================== **/
	/*| - |*/
	// Inverses each node(Triangle(-a, -b, -c))
	__dumb__ Triangle operator-()const;
	// Subtraction (Triangle(a - t.a, b - t.b, c - t.c))
	__dumb__ Triangle operator-(const Triangle &t)const;
	// Shifts triangle node by node
	__dumb__ Triangle& operator-=(const Triangle &t);
	// Same triangle, shifted by v
	__dumb__ Triangle operator-(const Vector3 &v)const;
	// Shifts triangle by v
	__dumb__ Triangle& operator-=(const Vector3 &v);

	/** ========================================================== **/
	/*| * |*/
	// Same triangle scaled by factor of f
	__dumb__ Triangle operator*(const float f)const;
	// t scaled by factor of f
	__dumb__ friend Triangle operator*(const float f, const Triangle &t);
	// Scales by factor of f
	__dumb__ Triangle& operator*=(const float f);
	// Same triangle, scaled by v (Triangle(a^v, b^v, c^v))
	__dumb__ Triangle operator*(const Vector3 &v)const;
	// Scales triangle by v
	__dumb__ Triangle& operator*=(const Vector3 &v);

	/** ========================================================== **/
	/*| / |*/
	// Same triangle, downscaled by a factor of f
	__dumb__ Triangle operator/(const float f)const;
	// Triangle, produced by inversing each node of t and multiplying by f
	__dumb__ friend Triangle operator/(const float f, const Triangle &t);
	// Downscales by a factor of f
	__dumb__ Triangle& operator/=(const float f);
	// Same triangle, downscaled by v (Triangle(a/v, b/v, c/v))
	__dumb__ Triangle operator/(const Vector3 &v)const;
	// Scales triangle by v
	__dumb__ Triangle& operator/=(const Vector3 &v);

	/** ========================================================== **/
	/*| <<=>> |*/
	// Transformed triangle
	__dumb__ Triangle operator>>(const Transform &t)const;
	// Transform a triangle
	__dumb__ Triangle& operator>>=(const Transform &t);
	// Detransformed triangle
	__dumb__ Triangle operator<<(const Transform &t)const;
	// Detransform a triangle
	__dumb__ Triangle& operator<<=(const Transform &t);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Functions: **/


	/** ========================================================== **/
	/*| Sorting |*/

	// Sorts the vertexes according to the given mases (acsending order)
	__dumb__ void sortByMases(float am, float bm, float cm);
	// Sorts the vertexes on an arbitrary axis (assuming, it's normalized)
	__dumb__ void sortOnAxis(const Vector3 &v);
	// Sorts the vertexes on an arbitrary axis (axis does not have to be normalized)
	__dumb__ void sortOnDirection(const Vector3 &v);
	// Sorts the vertexes on X axis
	__dumb__ void sortOnXaxis();
	// Sorts the vertexes on Y axis
	__dumb__ void sortOnYaxis();
	// Sorts the vertexes on Z axis
	__dumb__ void sortOnZaxis();


	/** ========================================================== **/
	/*| Surface |*/

	// Calculates the surface of the triangle
	__dumb__ float surface()const;


	/** ========================================================== **/
	/*| Normals |*/

	// Calculates the surface normal
	__dumb__ Vector3 normal()const;


	/** ========================================================== **/
	/*| Triangle space convertion |*/

	// Projects the vertex on the triangle plane
	__dumb__ Vector3 Triangle::projectVertex(const Vertex &v)const;
	// Calculates the mases of the vertexes, given the mass center (rv.x for a, rv.y for b, rv.z for c)
	// Notes:	This assumes, the vertex provided is on the same plane as the triangle itself;
	//			Sum of the returned value's components will allways be 1.
	__dumb__ Vector3 getMases(const Vertex &center)const;
	// Calculates the mases of the vertexes, given the mass center (rv.x for a, rv.y for b, rv.z for c)
	// Note: it's not nessessary for the center to be on the same plane with triangle
	__dumb__ Vector3 getMasesArbitrary(const Vertex &center)const;
	// Tells, if the point is inside the triangle, or not (assuming, it's on the same plane with the triangle)
	__dumb__ bool containsVertex(const Vertex &point)const;
	// Calculates the mass center of the triangle, if the sum of the components of the given vector of mases is 1
	__dumb__ Vertex massCenter(const Vector3 &masses)const;
	// Calculates the mass center of the triangle, even if the sum of the components of the given vector of mases is not equal to 1
	__dumb__ Vertex massCenterArbitrary(const Vector3 &masses)const;
	// Calculates the mass center of the triangle (vertex mases will be considered equal)
	__dumb__ Vertex massCenter()const;

	
	/** ========================================================== **/
	/*| casting |*/

	// Casts the Ray on the Triangle and tells, if the hit occures or not (if yes, hitDistance and hitPoint will be set to whatever they are)
	__dumb__ bool cast(const Ray &ray, float &hitDistance, Vertex &hitPoint, bool clipBackface)const;
};





#include"Triangle.impl.h"
