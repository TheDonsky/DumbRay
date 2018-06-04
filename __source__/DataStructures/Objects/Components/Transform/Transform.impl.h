#include"Transform.h"


/** -------------------------------------------------------------------------- **/
/** Construction: **/
// Constructor
__device__ __host__ inline Transform::Transform(const Vector3 &pos, const Vector3 &euler, const Vector3 &scl){
	position = pos;
	eulerAngles = euler;
	scale = scl;
	rebake();
}





/** -------------------------------------------------------------------------- **/
/** Getters: **/
// Transform position
__device__ __host__ inline const Vector3& Transform::getPosition()const{
	return(position);
}
// Transform euler angles
__device__ __host__ inline const Vector3& Transform::getEulerAngles()const{
	return(eulerAngles);
}
// Transform scale
__device__ __host__ inline const Vector3& Transform::getScale()const{
	return(scale);
}





/** -------------------------------------------------------------------------- **/
/** Setters: **/
// Sets tranform position
__device__ __host__ inline Transform& Transform::setPosition(const Vector3 &newPos){
	position = newPos;
	return(*this);
}
// Moves transform (translates position)
__device__ __host__ inline Transform& Transform::move(const Vector3 &posDelta){
	position += posDelta;
	return(*this);
}

// Sets transform euler angles
__device__ __host__ inline Transform& Transform::setEulerAngles(const Vector3 &newEuler){
	eulerAngles = newEuler;
	rebake();
	return(*this);
}
// Rotates transform
__device__ __host__ inline Transform& Transform::rotate(const Vector3 &eulerDelta){
	eulerAngles += eulerDelta;
	rebake();
	return(*this);
}

// Sets transform scale
__device__ __host__ inline Transform& Transform::setScale(const Vector3 &newScl){
	scale = newScl;
	rebake();
	return(*this);
}
// Scales transform
__device__ __host__ inline Transform& Transform::upscale(const Vector3 &sclDelta){
	scale += sclDelta;
	rebake();
	return(*this);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Transform & detransform: **/

/** ========================================================== **/
/*| Vectors |*/
// Transforms Vertex
__device__ __host__ inline Vector3& operator>>=(Vector3 &v, const Transform &t){
	return(v(v*t.trans.x + t.position.x, v*t.trans.y + t.position.y, v*t.trans.z + t.position.z));
}
// Transformed Vertex
__device__ __host__ inline Vector3 operator>>(const Vector3 &v, const Transform &t){
	return(Vector3(v*t.trans.x + t.position.x, v*t.trans.y + t.position.y, v*t.trans.z + t.position.z));
}

// Detransforms Vertex
__device__ __host__ inline Vector3& operator<<=(Vector3 &v, const Transform &t){
	Vector3 delta = v - t.position;
	return(v(delta*t.detrans.x, delta*t.detrans.y, delta*t.detrans.z));
}
// Detransformed Vertex
__device__ __host__ inline Vector3 operator<<(const Vector3 &v, const Transform &t){
	Vector3 delta = v - t.position;
	return(Vector3(delta*t.detrans.x, delta*t.detrans.y, delta*t.detrans.z));
}

/** ========================================================== **/
/*| Rays |*/
// Transforms Ray
__device__ __host__ inline Ray operator>>(const Ray &r, const Transform &trans) {
	return Ray(r.origin >> trans, trans.direction(r.direction));
}
// Transformed Ray
__device__ __host__ inline Ray& operator>>=(Ray &r, const Transform &trans) {
	r.origin >>= trans;
	r.direction = trans.direction(r.direction);
	return r;
}
// Detransforms Ray
__device__ __host__ inline Ray operator<<(const Ray &r, const Transform &trans) {
	return Ray(r.origin << trans, Vector3(r.direction * trans.detrans.x, r.direction * trans.detrans.y, r.direction * trans.detrans.z));
}
// Detransformed Ray
__device__ __host__ inline Ray& operator<<=(Ray &r, const Transform &trans) {
	r.origin <<= trans;
	r.direction(r.direction * trans.detrans.x, r.direction * trans.detrans.y, r.direction * trans.detrans.z);
	return r;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Directions: **/

/** ========================================================== **/
/*| Vectors |*/
// Forward
__device__ __host__ inline Vector3 Transform::front()const {
	return (trans.z / scale);
}
// Back
__device__ __host__ inline Vector3 Transform::back()const {
	return (-trans.z / scale);
}
// Right
__device__ __host__ inline Vector3 Transform::right()const {
	return (trans.x / scale);
}
// Left
__device__ __host__ inline Vector3 Transform::left()const {
	return (-trans.x / scale);
}
// Up
__device__ __host__ inline Vector3 Transform::up()const {
	return (trans.y / scale);
}
// Down
__device__ __host__ inline Vector3 Transform::down()const {
	return (-trans.y / scale);
}
// Direction
__device__ __host__ inline Vector3 Transform::direction(Vector3 dir)const {
	return ((dir.x * (trans.x / scale)) + (dir.y * (trans.y / scale)) + (dir.z * (trans.z / scale)));
}

/** ========================================================== **/
/*| Rays |*/
// Forward
__device__ __host__ inline Ray Transform::frontRay()const {
	return Ray(position, front());
}
// Back
__device__ __host__ inline Ray Transform::backRay()const {
	return Ray(position, back());
}
// Right
__device__ __host__ inline Ray Transform::rightRay()const {
	return Ray(position, right());
}
// Left
__device__ __host__ inline Ray Transform::leftRay()const {
	return Ray(position, left());
}
// Up
__device__ __host__ inline Ray Transform::upRay()const {
	return Ray(position, up());
}
// Down
__device__ __host__ inline Ray Transform::downRay()const {
	return Ray(position, down());
}
// Direction
__device__ __host__ inline Ray Transform::ray(Vector3 dir)const {
	return Ray(position, direction(dir));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** From/to Dson: **/
inline bool Transform::fromDson(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type() != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Transform can not be constructed from any Dson::Object other than Dson::Dict" << std::endl;
		return false;
	}
	const Dson::Dict &dict = (*((Dson::Dict*)(&object)));
	Transform transform;
	Vector3 vector;
	if (dict.contains("position")) {
		if (!vector.fromDson(dict["position"], errorStream)) return false;
		transform.setPosition(vector);
	}
	if (dict.contains("rotation")) {
		if (!vector.fromDson(dict["rotation"], errorStream)) return false;
		transform.setEulerAngles(vector);
	}
	if (dict.contains("scale")) {
		if (!vector.fromDson(dict["scale"], errorStream)) return false;
		transform.setScale(vector);
	}
	if (dict.contains("relative_position")) {
		if (!vector.fromDson(dict["relative_position"], errorStream)) return false;
		transform.move(
			(vector.x * transform.right()) + 
			(vector.y * transform.up()) + 
			(vector.z * transform.front()));
	}
	(*this) = transform;
	return true;
}
inline Dson::Dict Transform::toDson()const {
	Dson::Dict dict;
	dict.set("position", getPosition().toDson());
	dict.set("rotation", getEulerAngles().toDson());
	dict.set("scale", getScale().toDson());
	return dict;
}





/** -------------------------------------------------------------------------- **/
/** System rebake: **/
__device__ __host__ inline void Transform::rebake(){
	trans.x(1, 0, 0);
	trans.y(0, 1, 0);
	trans.z(0, 0, 1);

	Vector3 radianEuler = eulerAngles / RADIAN;

	trans.y.rotateTwoAxisAgainstThisRadian(trans.x, trans.z, -radianEuler.y);
	trans.x.rotateTwoAxisAgainstThisRadian(trans.y, trans.z, -radianEuler.x);
	trans.z.rotateTwoAxisAgainstThisRadian(trans.x, trans.y, -radianEuler.z);

	detrans.x.x = trans.x.x;
	detrans.x.y = trans.y.x;
	detrans.x.z = trans.z.x;
	detrans.y.x = trans.x.y;
	detrans.y.y = trans.y.y;
	detrans.y.z = trans.z.y;
	detrans.z.x = trans.x.z;
	detrans.z.y = trans.y.z;
	detrans.z.z = trans.z.z;

	trans.x ^= scale;
	trans.y ^= scale;
	trans.z ^= scale;

	detrans.x /= scale.x;
	detrans.y /= scale.y;
	detrans.z /= scale.z;
}
