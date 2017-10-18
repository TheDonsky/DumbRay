#include"Shapes.cuh"

namespace Shapes {
/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename Type1, typename Type2>
__dumb__ bool intersect(const Type1 &first, const Type2 &second) {
	return(first.intersects(second));
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename Type>
__dumb__ bool cast(const Ray &ray, const Type &object, bool clipBackface) {
	return object.cast(ray, clipBackface);
}

template<typename Type>
__dumb__ bool castPreInversed(const Ray &inversedRay, const Type &object, bool clipBackface) {
	return object.castPreInversed(inversedRay, clipBackface);
}

template<typename Type>
__dumb__ bool cast(const Ray &ray, const Type &object, float &hitDistance, Vertex &hitPoint, bool clipBackface) {
	return object.cast(ray, hitDistance, hitPoint, clipBackface);
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
//*
template<typename Type, typename BoundType>
__dumb__ bool sharePoint(const Type &a, const Type &b, const BoundType &commonPointBounds) {
	return a.sharesPoint(b, commonPointBounds);
}
//*/
template<typename Type1, typename Type2>
__dumb__ Vertex intersectionCenter(const Type1 &a, const Type2 &b) {
	return b.intersectionCenter(a);
}
template<typename Type1, typename Type2>
// Returns bounding box of intersection, if it exists, or some undefined value.
__dumb__ AABB intersectionBounds(const Type1 &a, const Type2 &b) {
	return b.intersectionBounds(a);
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename Type>
__dumb__ Vertex massCenter(const Type &shape) {
	return shape.massCenter();
}
template<typename Type>
__dumb__ AABB boundingBox(const Type &shape) {
	return shape.boundingBox();
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename Type>
__dumb__ void dump(const Type &shape) {
	shape.dump();
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** \\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\ **/
/** ########################################################################## **/
template<>
__dumb__ bool intersect<AABB, Triangle>(const AABB &box, const Triangle &tri){
	return box.intersects(tri);
}
template<>
__dumb__ bool intersect<Triangle, AABB>(const Triangle &tri, const AABB &box){
	return box.intersects(tri);
}

template<>
__dumb__ bool intersect<AABB, BakedTriFace>(const AABB &box, const BakedTriFace &tri){
	return box.intersects(tri.vert);
}
template<>
__dumb__ bool intersect<BakedTriFace, AABB>(const BakedTriFace &tri, const AABB &box){
	return box.intersects(tri.vert);
}

template<>
__dumb__ bool intersect<AABB, Vector3>(const AABB &box, const Vector3 &v){
	return box.contains(v);
}
template<>
__dumb__ bool intersect<Vector3, AABB>(const Vector3 &v, const AABB &box){
	return box.contains(v);
}

template<>
__dumb__ bool intersect<Triangle, Vector3>(const Triangle &tri, const Vector3 &v){
	return tri.containsVertex(v);
}
template<>
__dumb__ bool intersect<Vector3, Triangle>(const Vector3 &v, const Triangle &tri){
	return tri.containsVertex(v);
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<>
__dumb__ bool cast<AABB>(const Ray &ray, const AABB &box, bool clipBackface){
	register float castRes = box.cast(ray);
	return (castRes != FLT_MAX && ((!clipBackface) || (castRes > 0)));
}
template<>
__dumb__ bool cast<Triangle>(const Ray &ray, const Triangle &tri, bool clipBackface){
	float dist;
	Vector3 hitPoint;
	return tri.cast(ray, dist, hitPoint, clipBackface);
}
template<>
__dumb__ bool cast<BakedTriFace>(const Ray &ray, const BakedTriFace &tri, bool clipBackface){
	float dist;
	Vector3 hitPoint;
	return tri.vert.cast(ray, dist, hitPoint, clipBackface);
}

template<>
__dumb__ bool castPreInversed<AABB>(const Ray &inversedRay, const AABB &box, bool clipBackface){
	register float castRes = box.castPreInversed(inversedRay);
	return (castRes != FLT_MAX && ((!clipBackface) || (castRes > 0)));
}

template<>
__dumb__ bool cast<AABB>(const Ray &ray, const AABB &box, float &hitDistance, Vertex &hitPoint, bool clipBackface){
	register float castRes = box.cast(ray);
	if (castRes == FLT_MAX || (clipBackface && (castRes <= 0))) return false;
	hitDistance = castRes;
	hitPoint = ray.origin + (ray.direction * hitDistance);
	return true;
}
template<>
__dumb__ bool cast<Triangle>(const Ray &ray, const Triangle &tri, float &hitDistance, Vertex &hitPoint, bool clipBackface){
	return tri.cast(ray, hitDistance, hitPoint, clipBackface);
}
template<>
__dumb__ bool cast<BakedTriFace>(const Ray &ray, const BakedTriFace &tri, float &hitDistance, Vertex &hitPoint, bool clipBackface){
	return tri.vert.cast(ray, hitDistance, hitPoint, clipBackface);
}
template<>
__dumb__ bool cast<Vertex>(const Ray &ray, const Vertex &vert, float &hitDistance, Vertex &hitPoint, bool clipBackface){
	Vector3 delta = (vert - ray.origin);
	if ((delta * ray.direction) < -VECTOR_EPSILON) return false;
	if (delta.angleSin(ray.direction) < -VECTOR_EPSILON){
		hitDistance = delta.magnitude();
		hitPoint = vert;
		return true;
	}
	else return false;
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
//*
template<>
__dumb__ bool sharePoint<Triangle, AABB>(const Triangle &a, const Triangle &b, const AABB &commonPointBounds){
	if (commonPointBounds.contains(a.a) && (a.a == b.a || a.a == b.b || a.a == b.c)) return true;
	else if (commonPointBounds.contains(a.b) && (a.b == b.a || a.b == b.b || a.b == b.c)) return true;
	else return (commonPointBounds.contains(a.c) && (a.c == b.a || a.c == b.b || a.c == b.c));
}
template<>
__dumb__ bool sharePoint<BakedTriFace, AABB>(const BakedTriFace &a, const BakedTriFace &b, const AABB &commonPointBounds){
	if (commonPointBounds.contains(a.vert.a) && (a.vert.a == b.vert.a || a.vert.a == b.vert.b || a.vert.a == b.vert.c)) return true;
	else if (commonPointBounds.contains(a.vert.b) && (a.vert.b == b.vert.a || a.vert.b == b.vert.b || a.vert.b == b.vert.c)) return true;
	else return (commonPointBounds.contains(a.vert.c) && (a.vert.c == b.vert.a || a.vert.c == b.vert.b || a.vert.c == b.vert.c));
}
template<>
__dumb__ bool sharePoint<Vertex, AABB>(const Vertex &a, const Vertex &b, const AABB &commonPointBounds){
	return (commonPointBounds.contains(a) && a.isNearTo(b, VECTOR_EPSILON));
}
//*/
#define SHAPES_MIN(a, b) ((a < b) ? a : b)
#define SHAPES_MAX(a, b) ((a > b) ? a : b)
#define CROSS_POINT(name, from, to, fromV, toV, barrier) Vertex name = from + (to - from) * ((barrier - fromV) / (toV - fromV))
#define SHAPES_PUSH_INTERSECTIONS(front, back, sortFn, av, bv, cv, s, e) \
	for (int i = 0; i < front.size(); i++) { \
		Triangle &t = front[i]; \
		sortFn(); \
		if (cv < s) continue; /* a b c | | (1) */ \
		else if (av > e) continue; /* | | a b c (10) */ \
		else if (av <= s) { \
			CROSS_POINT(asc, t.a, t.c, av, cv, s); \
			if (bv <= s) { \
				CROSS_POINT(bsc, t.b, t.c, bv, cv, s); \
				if (cv <= e) back.push(Triangle(asc, bsc, t.c)); /* a b | c | (2) */ \
				else { /* a b | | c (3) */ \
					CROSS_POINT(bec, t.b, t.c, bv, cv, e); \
					back.push(Triangle(bsc, bec, asc)); \
					CROSS_POINT(aec, t.a, t.c, av, cv, e); \
					back.push(Triangle(asc, bec, aec)); \
				} \
			} \
			else if (bv <= e) { \
				if (cv <= e) { /* a | b c | (4) */ \
					back.push(Triangle(asc, t.b, t.c)); \
					CROSS_POINT(asb, t.a, t.b, av, bv, s); \
					back.push(Triangle(asc, asb, t.b)); \
				} \
				else { /* a | b | c (5) */ \
					CROSS_POINT(asb, t.a, t.b, av, bv, s); \
					CROSS_POINT(bec, t.b, t.c, bv, cv, e); \
					back.push(Triangle(asb, t.b, bec)); \
					back.push(Triangle(asc, asb, bec)); \
					CROSS_POINT(aec, t.a, t.c, av, cv, e); \
					back.push(Triangle(asc, bec, aec)); \
				} \
			} \
			else { /* a | | b c (6) */ \
				CROSS_POINT(asb, t.a, t.b, av, bv, s); \
				CROSS_POINT(aeb, t.a, t.b, av, bv, e); \
				back.push(Triangle(asc, asb, aeb)); \
				CROSS_POINT(aec, t.a, t.c, av, cv, e); \
				back.push(Triangle(asc, aeb, aec)); \
			} \
		} \
		else { \
			if (cv <= e) back.push(t); /* | a b c | (7) */ \
			else { \
				CROSS_POINT(aec, t.a, t.c, av, cv, e); \
				if (bv <= e) { /* | a b | c (8) */ \
					CROSS_POINT(bec, t.b, t.c, bv, cv, e); \
					back.push(Triangle(t.a, t.b, bec)); \
					back.push(Triangle(t.a, aec, bec)); \
				} \
				else { /* | a | b c (9) */ \
					CROSS_POINT(aeb, t.a, t.b, av, bv, e); \
					back.push(Triangle(t.a, aeb, aec)); \
				} \
			} \
		} \
	} \
	front.clear()
#define SHAPES_GET_INTERSECTING_TRIANGLES(name, boundingBox, triangleObject) \
	Stacktor<Triangle, 27> name; \
	{ \
		const Vertex start = boundingBox.getMin(); \
		const Vertex end = boundingBox.getMax(); \
		Stacktor<Triangle, 9> theListYouAreNotSupposedToNameYourResult; \
		theListYouAreNotSupposedToNameYourResult.push(triangleObject); \
		SHAPES_PUSH_INTERSECTIONS(theListYouAreNotSupposedToNameYourResult, name, t.sortOnXaxis, t.a.x, t.b.x, t.c.x, start.x, end.x); \
		SHAPES_PUSH_INTERSECTIONS(name, theListYouAreNotSupposedToNameYourResult, t.sortOnYaxis, t.a.y, t.b.y, t.c.y, start.y, end.y); \
		SHAPES_PUSH_INTERSECTIONS(theListYouAreNotSupposedToNameYourResult, name, t.sortOnZaxis, t.a.z, t.b.z, t.c.z, start.z, end.z); \
	}
template<>
__dumb__ AABB intersectionBounds<AABB, Triangle>(const AABB &aabb, const Triangle &triangle) {
	SHAPES_GET_INTERSECTING_TRIANGLES(list, aabb, triangle);
	if (list.size() > 0) {
		Vertex minimal = aabb.getMax();
		Vertex maximal = aabb.getMin();
		for (int i = 0; i < list.size(); i++) {
			const Triangle &t = list[i];
			maximal(
				SHAPES_MAX(SHAPES_MAX(SHAPES_MAX(maximal.x, t.a.x), t.b.x), t.c.x),
				SHAPES_MAX(SHAPES_MAX(SHAPES_MAX(maximal.y, t.a.y), t.b.y), t.c.y),
				SHAPES_MAX(SHAPES_MAX(SHAPES_MAX(maximal.z, t.a.z), t.b.z), t.c.z));
			minimal(
				SHAPES_MIN(SHAPES_MIN(SHAPES_MIN(minimal.x, t.a.x), t.b.x), t.c.x),
				SHAPES_MIN(SHAPES_MIN(SHAPES_MIN(minimal.y, t.a.y), t.b.y), t.c.y),
				SHAPES_MIN(SHAPES_MIN(SHAPES_MIN(minimal.z, t.a.z), t.b.z), t.c.z));
		}
		return AABB(minimal, maximal);
	}
	else return aabb;
}
template<>
__dumb__ Vertex intersectionCenter<AABB, Triangle>(const AABB &aabb, const Triangle &triangle) {
	//*
	SHAPES_GET_INTERSECTING_TRIANGLES(list, aabb, triangle);
	if (list.size() > 0) {
		float size = 0.0f;
		Vertex sum(0.0f, 0.0f, 0.0f);
		for (int i = 0; i < list.size(); i++) {
			const Triangle &t = list[i];
			float surface = ((128.0f * t.surface()) + 1.0f);
			sum += (t.massCenter() * surface);
			size += surface;
		}
		return (sum / size);

	}
	else return (aabb.getMin() - aabb.getMax());
	/*/
	return intersectionBounds<AABB, Triangle>(aabb, triangle).getCenter();
	//*/
}
#undef SHAPES_GET_INTERSECTING_TRIANGLES
#undef SHAPES_PUSH_INTERSECTIONS
#undef CROSS_POINT
template<>
__dumb__ Vertex intersectionCenter<AABB, BakedTriFace>(const AABB &aabb, const BakedTriFace &triangle) {
	return intersectionCenter<AABB, Triangle>(aabb, triangle.vert);
}
template<>
__dumb__ AABB intersectionBounds<AABB, BakedTriFace>(const AABB &aabb, const BakedTriFace &triangle) {
	return intersectionBounds<AABB, Triangle>(aabb, triangle.vert);
}
template<>
__dumb__ Vertex intersectionCenter<AABB, Vertex>(const AABB &aabb, const Vertex &vertex) {
	/*const Vertex start = aabb.getMin();
	const Vertex end = aabb.getMax();
	const Vertex vertStart = (vertex - EPSILON_VECTOR);
	const Vertex vertEnd = (vertex - EPSILON_VECTOR);
	return AABB(
		Vertex(
			(start.x < vertStart.x) ? vertStart.x : start.x, 
			(start.y < vertStart.y) ? vertStart.y : start.y, 
			(start.z < vertStart.z) ? vertStart.z : start.z), 
		Vertex(
			(end.x > vertEnd.x) ? vertEnd.x : end.x,
			(end.y > vertEnd.y) ? vertEnd.y : end.y,
			(end.z > vertEnd.z) ? vertEnd.z : end.z));*/
	return vertex;
}
template<>
__dumb__ AABB intersectionBounds<AABB, Vertex>(const AABB &aabb, const Vertex &vertex) {
	return AABB(vertex - EPSILON_VECTOR, vertex + EPSILON_VECTOR);
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<>
__dumb__ Vertex massCenter<Vertex>(const Vertex &shape){
	return shape;
}
template<>
__dumb__ AABB boundingBox<Vertex>(const Vertex &shape){
	return AABB(shape - EPSILON_VECTOR, shape + EPSILON_VECTOR);
}
template<>
__dumb__ Vertex massCenter<AABB>(const AABB &shape){
	return shape.getCenter();
}
template<>
__dumb__ AABB boundingBox<AABB>(const AABB &shape){
	return shape;
}
template<>
__dumb__ Vertex massCenter<Triangle>(const Triangle &shape){
	return shape.massCenter();
}
template<>
__dumb__ AABB boundingBox<Triangle>(const Triangle &shape){
	const Vertex start = Vector3(
		SHAPES_MIN(SHAPES_MIN(shape.a.x, shape.b.x), shape.c.x),
		SHAPES_MIN(SHAPES_MIN(shape.a.y, shape.b.y), shape.c.y),
		SHAPES_MIN(SHAPES_MIN(shape.a.z, shape.b.z), shape.c.z));
	const Vertex end = Vector3(
		SHAPES_MAX(SHAPES_MAX(shape.a.x, shape.b.x), shape.c.x), 
		SHAPES_MAX(SHAPES_MAX(shape.a.y, shape.b.y), shape.c.y), 
		SHAPES_MAX(SHAPES_MAX(shape.a.z, shape.b.z), shape.c.z));
	return AABB(start, end);
}
#undef SHAPES_MIN
#undef SHAPES_MAX
template<>
__dumb__ Vertex massCenter<BakedTriFace>(const BakedTriFace &shape){
	return shape.vert.massCenter();
}
template<>
__dumb__ AABB boundingBox<BakedTriFace>(const BakedTriFace &shape){
	return boundingBox<Triangle>(shape.vert);
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<>
__dumb__ void dump<Vertex>(const Vertex &v){
	printf("(%.1f, %.1f, %.1f)", v.x, v.y, v.z);
}
template<>
__dumb__ void dump<Triangle>(const Triangle &tri){
	dump<Vertex>(tri.a);
	dump<Vertex>(tri.b);
	dump<Vertex>(tri.c);
}
template<>
__dumb__ void dump<BakedTriFace>(const BakedTriFace &face){
	printf("FACE:\n");
	printf("   VERTS: <"); dump<Triangle>(face.vert); printf(">\n");
	printf("   NORMS: <"); dump<Triangle>(face.norm); printf(">\n");
	printf("   TEXS:  <"); dump<Triangle>(face.tex); printf(">\n");
}

}

