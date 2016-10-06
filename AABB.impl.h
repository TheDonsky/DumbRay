#include"AABB.h"
#include"MemManip.cuh"


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Construction: **/
// Default constructor (does nothing)
__dumb__ AABB::AABB(){}
// Constructor (minimal and maximal boundaries will be defined by s and e)
__dumb__ AABB::AABB(const Vertex &s, const Vertex &e){
	start = s;
	end = e;
	fix();
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Getters and setters: **/


/** ========================================================== **/
/*| Getters |*/

// Returns the center
__dumb__ Vertex AABB::getCenter()const{
	return((start + end) * 0.5f);
}
// Returns the minimum boundary
__dumb__ Vertex AABB::getMin()const{
	return(start);
}
// Returns the maximum boundary
__dumb__ Vertex AABB::getMax()const{
	return(end);
}
// Returns the extents (dimensions against center(half of the total size))
__dumb__ Vector3 AABB::getExtents()const{
	return((end - start) * 0.5f);
}
// Returns the dimensions of the bounding box
__dumb__ Vector3 AABB::getSize()const{
	return(end - start);
}


/** ========================================================== **/
/*| Setters |*/

// Sets the center (moves both min and max, while reserving size)
__dumb__ void AABB::setCenter(const Vertex &v){
	Vector3 delta = v - getCenter();
	start += delta;
	end += delta;
}
// Sets the minimum boundary (maximum boundary is reserved, or swapped, if the resulting AABB runs a risc of having negative dimensions)
__dumb__ void AABB::setMin(const Vertex &v){
	start = v;
	fix();
}
// Sets the maximum boundary (minimum boundary is reserved, or swapped, if the resulting AABB runs a risc of having negative dimensions)
__dumb__ void AABB::setMax(const Vertex &v){
	end = v;
	fix();
}
// Sets the extents (dimensions against center(half of the total size); center is reserved)
__dumb__ void AABB::setExtents(const Vector3 &v){
	Vector3 c = getCenter();
	start = c - v;
	end = c + v;
	fix();
}
// Sets the dimensions of the bounding box (center is reserved)
__dumb__ void AABB::setSize(const Vector3 &v){
	setExtents(v * 0.5f);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Intersections: **/
// Checks, if the axis aligned bounding box contains the Vertex
__dumb__ bool AABB::contains(const Vertex &v)const{
	return(v.x >= start.x && v.y >= start.y && v.z >= start.z && v.x <= end.x && v.y <= end.y && v.z <= end.z);
}
template<>
__dumb__ bool AABB::intersectsTri<0>(Triangle t)const;
// Checks, if the axis aligned bounding box intersects the given triangle
__dumb__ bool AABB::intersects(const Triangle &t)const{
	return(intersectsTri<0>(t));
}
// Checks, if the axis aligned bounding box intersects the given triangle, defined by vertexes
__dumb__ bool AABB::intersectsTriangle(const Vertex &a, const Vertex &b, const Vertex &c)const{
	return(intersectsTri<0>(Triangle(a, b, c)));
}
// Returns the distance the ray needs to travel in order to hit the box (0 or negative if inside, FLT_MAX, if hit can't occure)
__dumb__ float AABB::cast(const Ray &r)const{
	return castPreInversed(Ray(r.origin, 1.0f / r.direction));
}
// Returns the distance the ray needs to travel in order to hit the box (ray should have pre-inversed direction vector(gives some speedup); 0 or negative if inside, FLT_MAX, if hit can't occure)
__dumb__ float AABB::castPreInversed(const Ray &r)const{
	register float ds = (start.x - r.origin.x) * r.direction.x;
	register float de = (end.x - r.origin.x) * r.direction.x;
	register float mn = min(ds, de), mx = max(ds, de);
	ds = (start.y - r.origin.y) * r.direction.y;
	de = (end.y - r.origin.y) * r.direction.y;
	mn = max(mn, min(ds, de));
	mx = min(mx, max(ds, de));
	ds = (start.z - r.origin.z) * r.direction.z;
	de = (end.z - r.origin.z) * r.direction.z;
	mn = max(mn, min(ds, de));
	mx = min(mx, max(ds, de));
	if (mn > mx + VECTOR_EPSILON) return FLT_MAX;
	return mn;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Private functions: **/
// checks, if start.x <= end.x, start.y <= end.y and start.z < end.z and swaps values if any of these requirements is not met
__dumb__ void AABB::fix(){
	if (start.x > end.x) MemManip::swap(start.x, end.x);
	if (start.y > end.y) MemManip::swap(start.y, end.y);
	if (start.z > end.z) MemManip::swap(start.z, end.z);
}
template<unsigned int dimm>
// checks the triangle intersection (if dimm is 0, for all 3 dimensions, if its 1, for x and y only and, in case of 2, just y)
__dumb__ bool AABB::intersectsTri(Triangle t)const{
	return true;
}
template<>
__dumb__ bool AABB::intersectsTri<0>(Triangle t)const{
	t.sortOnZaxis();
	return intersectsTri<0>(t, t.a.z, t.b.z, t.c.z, start.z, end.z);
}
template<>
__dumb__ bool AABB::intersectsTri<1>(Triangle t)const{
	t.sortOnXaxis();
	return intersectsTri<1>(t, t.a.x, t.b.x, t.c.x, start.x, end.x);
}
template<>
__dumb__ bool AABB::intersectsTri<2>(Triangle t)const{
	t.sortOnYaxis();
	return intersectsTri<2>(t, t.a.y, t.b.y, t.c.y, start.y, end.y);
}
#define CROSS_POINT(name, from, to, fromV, toV, barrier) Vertex name = from + (to - from) * ((barrier - fromV) / (toV - fromV))
template<unsigned int dimm>
__dumb__ bool AABB::intersectsTri(const Triangle &t, float av, float bv, float cv, float s, float e)const{
	if (cv < s) return(false); // a b c | | (1)
	if (av > e) return(false); // | | a b c (10)
	if (av <= s){
		CROSS_POINT(asc, t.a, t.c, av, cv, s);
		if (bv <= s){
			CROSS_POINT(bsc, t.b, t.c, bv, cv, s);
			if (cv <= e) return(intersectsTri<dimm + 1>(Triangle(asc, bsc, t.c))); // a b | c | (2)
			else{ // a b | | c (3)
				CROSS_POINT(bec, t.b, t.c, bv, cv, e);
				if (intersectsTri<dimm + 1>(Triangle(bsc, bec, asc))) return(true);
				CROSS_POINT(aec, t.a, t.c, av, cv, e);
				return(intersectsTri<dimm + 1>(Triangle(asc, bec, aec)));
			}
		}
		else if (bv <= e){
			if (cv <= e){ // a | b c | (4)
				if (intersectsTri<dimm + 1>(Triangle(asc, t.b, t.c))) return(true);
				CROSS_POINT(asb, t.a, t.b, av, bv, s);
				return(intersectsTri<dimm + 1>(Triangle(asc, asb, t.b)));
			}
			else{ // a | b | c (5)
				CROSS_POINT(asb, t.a, t.b, av, bv, s);
				CROSS_POINT(bec, t.b, t.c, bv, cv, e);
				if (intersectsTri<dimm + 1>(Triangle(asb, t.b, bec))) return(true);
				if (intersectsTri<dimm + 1>(Triangle(asc, asb, bec))) return(true);
				CROSS_POINT(aec, t.a, t.c, av, cv, e);
				return(intersectsTri<dimm + 1>(Triangle(asc, bec, aec)));
			}
		}
		else{ // a | | b c (6)
			CROSS_POINT(asb, t.a, t.b, av, bv, s);
			CROSS_POINT(aeb, t.a, t.b, av, bv, e);
			if (intersectsTri<dimm + 1>(Triangle(asc, asb, aeb))) return(true);
			CROSS_POINT(aec, t.a, t.c, av, cv, e);
			return(intersectsTri<dimm + 1>(Triangle(asc, aeb, aec)));
		}
	}
	else{
		if (cv <= e) return(intersectsTri<dimm + 1>(t)); // | a b c | (7)
		else{
			CROSS_POINT(aec, t.a, t.c, av, cv, e);
			if (bv <= e){ // | a b | c (8)
				CROSS_POINT(bec, t.b, t.c, bv, cv, e);
				if (intersectsTri<dimm + 1>(Triangle(t.a, t.b, bec))) return(true);
				return(intersectsTri<dimm + 1>(Triangle(t.a, aec, bec)));
			}
			else{ // | a | b c (9)
				CROSS_POINT(aeb, t.a, t.b, av, bv, e);
				return(intersectsTri<dimm + 1>(Triangle(t.a, aeb, aec)));
			}
		}
	}
}
#undef CROSS_POINT



