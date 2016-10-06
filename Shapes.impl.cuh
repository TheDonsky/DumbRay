#include"Shapes.cuh"



/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<>
__dumb__ bool Shapes::intersect<AABB, Triangle>(const AABB &box, const Triangle &tri){
	return box.intersects(tri);
}
template<>
__dumb__ bool Shapes::intersect<Triangle, AABB>(const Triangle &tri, const AABB &box){
	return box.intersects(tri);
}

template<>
__dumb__ bool Shapes::intersect<AABB, BakedTriFace>(const AABB &box, const BakedTriFace &tri){
	return box.intersects(tri.vert);
}
template<>
__dumb__ bool Shapes::intersect<BakedTriFace, AABB>(const BakedTriFace &tri, const AABB &box){
	return box.intersects(tri.vert);
}

template<>
__dumb__ bool Shapes::intersect<AABB, Vector3>(const AABB &box, const Vector3 &v){
	return box.contains(v);
}
template<>
__dumb__ bool Shapes::intersect<Vector3, AABB>(const Vector3 &v, const AABB &box){
	return box.contains(v);
}

template<>
__dumb__ bool Shapes::intersect<Triangle, Vector3>(const Triangle &tri, const Vector3 &v){
	return tri.containsVertex(v);
}
template<>
__dumb__ bool Shapes::intersect<Vector3, Triangle>(const Vector3 &v, const Triangle &tri){
	return tri.containsVertex(v);
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<>
__dumb__ bool Shapes::cast<AABB>(const Ray &ray, const AABB &box, bool clipBackface){
	register float castRes = box.cast(ray);
	return (castRes != FLT_MAX && ((!clipBackface) || (castRes > 0)));
}
template<>
__dumb__ bool Shapes::cast<Triangle>(const Ray &ray, const Triangle &tri, bool clipBackface){
	float dist;
	Vector3 hitPoint;
	return tri.cast(ray, dist, hitPoint, clipBackface);
}
template<>
__dumb__ bool Shapes::cast<BakedTriFace>(const Ray &ray, const BakedTriFace &tri, bool clipBackface){
	float dist;
	Vector3 hitPoint;
	return tri.vert.cast(ray, dist, hitPoint, clipBackface);
}

template<>
__dumb__ bool Shapes::castPreInversed<AABB>(const Ray &inversedRay, const AABB &box, bool clipBackface){
	register float castRes = box.castPreInversed(inversedRay);
	return (castRes != FLT_MAX && ((!clipBackface) || (castRes > 0)));
}

template<>
__dumb__ bool Shapes::cast<AABB>(const Ray &ray, const AABB &box, float &hitDistance, Vertex &hitPoint, bool clipBackface){
	register float castRes = box.cast(ray);
	if (castRes == FLT_MAX || (clipBackface && (castRes <= 0))) return false;
	hitDistance = castRes;
	hitPoint = ray.origin + (ray.direction * hitDistance);
	return true;
}
template<>
__dumb__ bool Shapes::cast<Triangle>(const Ray &ray, const Triangle &tri, float &hitDistance, Vertex &hitPoint, bool clipBackface){
	return tri.cast(ray, hitDistance, hitPoint, clipBackface);
}
template<>
__dumb__ bool Shapes::cast<BakedTriFace>(const Ray &ray, const BakedTriFace &tri, float &hitDistance, Vertex &hitPoint, bool clipBackface){
	return tri.vert.cast(ray, hitDistance, hitPoint, clipBackface);
}
template<>
__dumb__ bool Shapes::cast<Vertex>(const Ray &ray, const Vertex &vert, float &hitDistance, Vertex &hitPoint, bool clipBackface){
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
template<>
__dumb__ bool Shapes::sharePoint<Triangle, AABB>(const Triangle &a, const Triangle &b, const AABB &commonPointBounds){
	if (commonPointBounds.contains(a.a) && (a.a == b.a || a.a == b.b || a.a == b.c)) return true;
	else if (commonPointBounds.contains(a.b) && (a.b == b.a || a.b == b.b || a.b == b.c)) return true;
	else return (commonPointBounds.contains(a.c) && (a.c == b.a || a.c == b.b || a.c == b.c));
}
template<>
__dumb__ bool Shapes::sharePoint<BakedTriFace, AABB>(const BakedTriFace &a, const BakedTriFace &b, const AABB &commonPointBounds){
	if (commonPointBounds.contains(a.vert.a) && (a.vert.a == b.vert.a || a.vert.a == b.vert.b || a.vert.a == b.vert.c)) return true;
	else if (commonPointBounds.contains(a.vert.b) && (a.vert.b == b.vert.a || a.vert.b == b.vert.b || a.vert.b == b.vert.c)) return true;
	else return (commonPointBounds.contains(a.vert.c) && (a.vert.c == b.vert.a || a.vert.c == b.vert.b || a.vert.c == b.vert.c));
}
template<>
__dumb__ bool Shapes::sharePoint<Vertex, AABB>(const Vertex &a, const Vertex &b, const AABB &commonPointBounds){
	return (commonPointBounds.contains(a) && a.isNearTo(b, VECTOR_EPSILON));
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<>
__dumb__ Vertex Shapes::massCenter<Vertex>(const Vertex &shape){
	return shape;
}
template<>
__dumb__ AABB Shapes::boundingBox<Vertex>(const Vertex &shape){
	return AABB(shape - EPSILON_VECTOR, shape + EPSILON_VECTOR);
}
template<>
__dumb__ Vertex Shapes::massCenter<AABB>(const AABB &shape){
	return shape.getCenter();
}
template<>
__dumb__ AABB Shapes::boundingBox<AABB>(const AABB &shape){
	return shape;
}
template<>
__dumb__ Vertex Shapes::massCenter<Triangle>(const Triangle &shape){
	return shape.massCenter();
}
template<>
__dumb__ AABB Shapes::boundingBox<Triangle>(const Triangle &shape){
	const Vertex start = Vector3(min(min(shape.a.x, shape.b.x), shape.c.x), min(min(shape.a.y, shape.b.y), shape.c.y), min(min(shape.a.z, shape.b.z), shape.c.z));
	const Vertex end = Vector3(max(max(shape.a.x, shape.b.x), shape.c.x), max(max(shape.a.y, shape.b.y), shape.c.y), max(max(shape.a.z, shape.b.z), shape.c.z));
	return AABB(start, end);
}
template<>
__dumb__ Vertex Shapes::massCenter<BakedTriFace>(const BakedTriFace &shape){
	return shape.vert.massCenter();
}
template<>
__dumb__ AABB Shapes::boundingBox<BakedTriFace>(const BakedTriFace &shape){
	return boundingBox<Triangle>(shape.vert);
}


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<>
__dumb__ void Shapes::dump<Vertex>(const Vertex &v){
	printf("(%.1f, %.1f, %.1f)", v.x, v.y, v.z);
}
template<>
__dumb__ void Shapes::dump<Triangle>(const Triangle &tri){
	Shapes::dump<Vertex>(tri.a);
	Shapes::dump<Vertex>(tri.b);
	Shapes::dump<Vertex>(tri.c);
}
template<>
__dumb__ void Shapes::dump<BakedTriFace>(const BakedTriFace &face){
	printf("FACE:\n");
	printf("   VERTS: <"); Shapes::dump<Triangle>(face.vert); printf(">\n");
	printf("   NORMS: <"); Shapes::dump<Triangle>(face.norm); printf(">\n");
	printf("   TEXS:  <"); Shapes::dump<Triangle>(face.tex); printf(">\n");
}
