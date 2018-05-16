#include"Triangle.h"
#include"../../../../Namespaces/MemManip/MemManip.cuh"


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Construction: **/
// Does nothing
__dumb__ Triangle::Triangle(){}
// Simple constructor
__dumb__ Triangle::Triangle(const Vertex &A, const Vertex &B, const Vertex &C){
	a = A;
	b = B;
	c = C;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Operators: **/


/** ========================================================== **/
/*| + |*/
// Same as t
__dumb__ Triangle Triangle::operator+()const {
	return (*this);
}
// Sum (Triangle(a + t.a, b + t.b, c +_ t.c))
__dumb__ Triangle Triangle::operator+(const Triangle &t)const {
	return Triangle(a + t.a, b + t.b, c + t.c);
}
// Shifts triangle node by node
__dumb__ Triangle& Triangle::operator+=(const Triangle &t) {
	a += t.a;
	b += t.b;
	c += t.c;
	return(*this);
}
// Same Triangle, shifted by v
__dumb__ Triangle Triangle::operator+(const Vector3 &v)const {
	return Triangle(a + v, b + v, c + v);
}
// Shifts Triangle by v
__dumb__ Triangle& Triangle::operator+=(const Vector3 &v) {
	a += v;
	b += v;
	c += v;
	return (*this);
}

/** ========================================================== **/
/*| - |*/
// Inverses each node(Triangle(-a, -b, -c))
__dumb__ Triangle Triangle::operator-()const {
	return Triangle(-a, -b, -c);
}
// Subtraction (Triangle(a - t.a, b - t.b, c - t.c))
__dumb__ Triangle Triangle::operator-(const Triangle &t)const {
	return Triangle(a - t.a, b - t.b, c - t.c);
}
// Shifts triangle node by node
__dumb__ Triangle& Triangle::operator-=(const Triangle &t) {
	a -= t.a;
	b -= t.b;
	c -= t.c;
	return (*this);
}
// Same triangle, shifted by v
__dumb__ Triangle Triangle::operator-(const Vector3 &v)const {
	return Triangle(a - v, b - v, c - v);
}
// Shifts triangle by v
__dumb__ Triangle& Triangle::operator-=(const Vector3 &v) {
	a -= v;
	b -= v;
	c -= v;
	return (*this);
}

/** ========================================================== **/
/*| * |*/
// Same triangle scaled by factor of f
__dumb__ Triangle Triangle::operator*(const float f)const {
	return Triangle(a * f, b * f, c * f);
}
// t scaled by factor of f
__dumb__ Triangle operator*(const float f, const Triangle &t) {
	return (t * f);
}
// Scales by factor of f
__dumb__ Triangle& Triangle::operator*=(const float f) {
	a *= f;
	b *= f;
	c *= f;
	return (*this);
}
// Same triangle, scaled by v (Triangle(a^v, b^v, c^v))
__dumb__ Triangle Triangle::operator*(const Vector3 &v)const {
	return Triangle(a^v, b^v, c^v);
}
// Scales triangle by v
__dumb__ Triangle& Triangle::operator*=(const Vector3 &v) {
	a ^= v;
	b ^= v;
	c ^= v;
	return (*this);
}

/** ========================================================== **/
/*| / |*/
// Same triangle, downscaled by a factor of f
__dumb__ Triangle Triangle::operator/(const float f)const {
	register float fInv = 1.0f / f;
	return Triangle(a * fInv, b * fInv, c * fInv);
}
// Triangle, produced by inversing each node of t and multiplying by f
__dumb__ Triangle operator/(const float f, const Triangle &t) {
	return Triangle(f / t.a, f / t.b, f / t.c);
}
// Downscales by a factor of f
__dumb__ Triangle& Triangle::operator/=(const float f) {
	register float fInv = 1.0f / f;
	a *= fInv;
	b *= fInv;
	c *= fInv;
	return (*this);
}
// Same triangle, downscaled by v (Triangle(a/v, b/v, c/v))
__dumb__ Triangle Triangle::operator/(const Vector3 &v)const {
	return Triangle(a / v, b / v, c / v);
}
// Scales triangle by v
__dumb__ Triangle& Triangle::operator/=(const Vector3 &v) {
	a /= v;
	b /= v;
	c /= v;
	return (*this);
}

/** ========================================================== **/
/*| <<=>> |*/
// Transformed triangle
__dumb__ Triangle Triangle::operator>>(const Transform &t)const{
	return Triangle(a >> t, b >> t, c >> t);
}
// Transform a triangle
__dumb__ Triangle& Triangle::operator>>=(const Transform &t) {
	a >>= t;
	b >>= t;
	c >>= t;
	return (*this);
}
// Detransformed triangle
__dumb__ Triangle Triangle::operator<<(const Transform &t)const {
	return Triangle(a << t, b << t, c << t);
}
// Detransform a triangle
__dumb__ Triangle& Triangle::operator<<=(const Transform &t) {
	a <<= t;
	b <<= t;
	c <<= t;
	return (*this);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Functions: **/


/** ========================================================== **/
/*| Sorting |*/

// Sorts the vertexes according to the given masses (acsending order)
__dumb__ void Triangle::sortByMases(float am, float bm, float cm){
	if (am > bm){
		if (cm > am) MemManip::swap(a, b); // b a c
		else{
			if (bm > cm) MemManip::swap(a, c); // c b a
			else{ Vertex tmp = a; a = b; b = c; c = tmp; } // b c a
		}
	}
	else{
		if (am > cm){ Vertex tmp = a; a = c; c = b; b = tmp; } // c a b
		else if (bm > cm) MemManip::swap(b, c); // a c b
	}
}
// Sorts the vertexes on an arbitrary axis (assuming, it's normalized)
__dumb__ void Triangle::sortOnAxis(const Vector3 &v){
	sortByMases(a.normalLengthOnUnitVector(v), b.normalLengthOnUnitVector(v), c.normalLengthOnUnitVector(v));
}
// Sorts the vertexes on an arbitrary axis (axis does not have to be normalized)
__dumb__ void Triangle::sortOnDirection(const Vector3 &v){
	sortOnAxis(v.normalized());
}
// Sorts the vertexes on X axis
__dumb__ void Triangle::sortOnXaxis(){
	sortByMases(a.x, b.x, c.x);
}
// Sorts the vertexes on Y axis
__dumb__ void Triangle::sortOnYaxis(){
	sortByMases(a.y, b.y, c.y);
}
// Sorts the vertexes on Z axis
__dumb__ void Triangle::sortOnZaxis(){
	sortByMases(a.z, b.z, c.z);
}


/** ========================================================== **/
/*| Surface |*/

// Calculates the surface of the triangle
__dumb__ float Triangle::surface()const{
	return(((b - a) & (c - a)).magnitude() * 0.5f);
}


/** ========================================================== **/
/*| Normals |*/

// Calculates the surface normal
__dumb__ Vector3 Triangle::normal()const {
	return ((c - a) & (b - a));
}


/** ========================================================== **/
/*| Triangle space convertion |*/

// Projects the vertex on the triangle plane
__dumb__ Vector3 Triangle::projectVertex(const Vertex &v)const{
	return(v + (a - v).normalOn((b - a) & (c - a)));
}
// Calculates the masses of the vertexes, given the mass center (rv.x for a, rv.y for b, rv.z for c)
// Notes:	This assumes, the vertex provided is on the same plane as the triangle itself;
//			Sum of the returned value's components will allways be 1.
__dumb__ Vector3 Triangle::getMasses(const Vertex &center)const{
	Vector3 ab = b - a;
	Vector3 bc = c - b;
	Vector3 ae = ab - ab.normalOn(bc);

	Vector3 ax = (center - a);
	Vector3 ad = ax.normalOn(ae);
	if (ae.x < 0){ ad.x = -ad.x; ae.x = -ae.x; }
	if (ae.y < 0){ ad.y = -ad.y; ae.y = -ae.y; }
	if (ae.z < 0){ ad.z = -ad.z; ae.z = -ae.z; }
	float div = ad.x + ad.y + ad.z;
	if (div == 0) return(Vector3(1, 0, 0));
	float g = (ae.x + ae.y + ae.z) / div;

	float t;
	Vector3 by = a + ax * g - b;
	if (bc.x < 0){ bc.x = -bc.x; by.x = -by.x; }
	if (bc.y < 0){ bc.y = -bc.y; by.y = -by.y; }
	if (bc.z < 0){ bc.z = -bc.z; by.z = -by.z; }
	div = bc.x + bc.y + bc.z;
	if (div == 0) t = 0;
	else t = (by.x + by.y + by.z) / div;

	float cc = t;
	float bb = (1 - t);
	float aa = (g - 1);

	return(Vector3(aa, bb, cc) / (aa + bb + cc));
}
// Calculates the masses of the vertexes, given the mass center (rv.x for a, rv.y for b, rv.z for c)
// Note: it's not nessessary for the center to be on the same plane with triangle
__dumb__ Vector3 Triangle::getMassesArbitrary(const Vertex &center)const{
	return(getMasses(projectVertex(center)));
}
// Tells, if the point is inside the triangle, or not (assuming, it's on the same plane with the triangle)
__dumb__ bool Triangle::containsVertex(const Vertex &point)const{
	/*
	// This is fast, but produces some artifacts under some sircumstances..
	register Vector3 xa = (a - point);
	register Vector3 xb = (b - point);
	register Vector3 xc = (c - point);
	return ((xa & xb).magnitude() + (xb & xc).magnitude() + (xc & xa).magnitude() <= ((b - a) & (c - a)).magnitude() + 192 * VECTOR_EPSILON);
	/*/
	// This is reliable, but rather slow
	if (a == b || b == c || c == a) return false;
	if (point == a || point == b || point == c) return true;
	register Vector3 ab = (b - a);
	register Vector3 bc = (c - b);
	register Vector3 ca = (a - c);
	register Vector3 ax = (point - a);
	register Vector3 bx = (point - b);
	register Vector3 cx = (point - c);
	return((ab * ax) / ax.magnitude() + 32 * VECTOR_EPSILON >= -(ab * ca) / ca.magnitude()
		&& (bc * bx) / bx.magnitude() + 32 * VECTOR_EPSILON >= -(bc * ab) / ab.magnitude()
		&& (ca * cx) / cx.magnitude() + 32 * VECTOR_EPSILON >= -(ca * bc) / bc.magnitude());
	/*/
	
	register Vector3 center = massCenter();
	if (point == center) return true;
	register Vector3 p = point + (center - point).normalize() * (64 * VECTOR_EPSILON);
	Vector3 ab = (b - a);
	Vector3 ax = (p - a);
	Vector3 hitNormal = (ax & ab);
	if (hitNormal.sqrMagnitude() < VECTOR_EPSILON * VECTOR_EPSILON)
		return (ax * (p - b) < 16 * VECTOR_EPSILON);
	else return (((hitNormal * ((p - b) & (c - b))) > -VECTOR_EPSILON) && ((hitNormal * ((p - c) & (a - c)) > -VECTOR_EPSILON)));
	//*/
}
// Calculates the mass center of the triangle, if the sum of the components of the given vector of masses is 1
__dumb__ Vertex Triangle::massCenter(const Vector3 &masses)const{
	return(a * masses.x + b*masses.y + c*masses.z);
}
// Calculates the mass center of the triangle, even if the sum of the components of the given vector of masses is not equal to 1
__dumb__ Vertex Triangle::massCenterArbitrary(const Vector3 &masses)const{
	return(massCenter(masses) / (masses.x + masses.y + masses.z));
}
// Calculates the mass center of the triangle (vertex masses will be considered equal)
__dumb__ Vertex Triangle::massCenter()const{
	return((a + b + c) / 3);
}


/** ========================================================== **/
/*| casting |*/

// Casts the Ray on the Triangle and tells, if the hit occures or not (if yes, hitDistance and hitPoint will be set to whatever they are)
__dumb__ bool Triangle::cast(const Ray &ray, float &hitDistance, Vertex &hitPoint, bool clipBackface)const{
	register Vector3 normal = ((b - a) & (c - a));
	register float deltaProjection = ((a - ray.origin) * normal);
	if (clipBackface && (deltaProjection < -VECTOR_EPSILON)) return false;
	register float dirProjection = (ray.direction * normal);
	if (deltaProjection * dirProjection <= 0) return false;
	register float distance = deltaProjection / dirProjection;
	Vertex hitVert = ray.origin + ray.direction * distance;
	if (containsVertex(hitVert)){
		hitDistance = distance;
		hitPoint = hitVert;
		return true;
	}
	else return false;
}
