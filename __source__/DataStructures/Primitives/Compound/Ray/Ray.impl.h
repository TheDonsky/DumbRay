#include"Ray.h"

/* -------------------------------------------------------------------------- */
// Macros:
#define SetAndReturnThis(orgn, dir) origin = orgn; direction = dir; return(*this)





/** -------------------------------------------------------------------------- **/
/** Construction: **/
__device__ __host__ inline Ray::Ray(){}
__device__ __host__ inline Ray::Ray(const Vector3 &org, const Vector3 &dir){
	origin = org;
	direction = dir;
}
__device__ __host__ inline Ray::Ray(const Ray &r){
	origin = r.origin;
	direction = r.direction;
}
__device__ __host__ inline Ray& Ray::operator()(const Vector3 &org, const Vector3 &dir){
	SetAndReturnThis(org, dir);
}





/** -------------------------------------------------------------------------- **/
/** Stream operators: **/
inline static std::istream& operator>>(std::istream &stream, Ray &r){
	return(stream >> r.origin >> r.direction);
}
inline static std::ostream& operator<<(std::ostream &stream, const Ray &r){
	return(stream << "(Origin: " << r.origin << "; Direction: " << r.direction << ")");
}





/** -------------------------------------------------------------------------- **/
/** Macro undefs: **/
#undef SetAndReturnThis
