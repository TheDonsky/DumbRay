#include"Transform.h"


/** -------------------------------------------------------------------------- **/
/** Construction: **/
__device__ __host__ inline Transform::Transform(const Vector3 &pos, const Vector3 &euler, const Vector3 &scl){
	position = pos;
	eulerAngles = euler;
	scale = scl;
	rebake();
}





/** -------------------------------------------------------------------------- **/
/** Getters: **/
__device__ __host__ inline const Vector3& Transform::getPosition()const{
	return(position);
}
__device__ __host__ inline const Vector3& Transform::getEulerAngles()const{
	return(eulerAngles);
}
__device__ __host__ inline const Vector3& Transform::getScale()const{
	return(scale);
}





/** -------------------------------------------------------------------------- **/
/** Setters: **/
__device__ __host__ inline Transform& Transform::setPosition(const Vector3 &newPos){
	position = newPos;
	return(*this);
}
__device__ __host__ inline Transform& Transform::move(const Vector3 &posDelta){
	position += posDelta;
	return(*this);
}

__device__ __host__ inline Transform& Transform::setEulerAngles(const Vector3 &newEuler){
	eulerAngles = newEuler;
	rebake();
	return(*this);
}
__device__ __host__ inline Transform& Transform::rotate(const Vector3 &eulerDelta){
	eulerAngles += eulerDelta;
	rebake();
	return(*this);
}

__device__ __host__ inline Transform& Transform::setScale(const Vector3 &newScl){
	scale = newScl;
	rebake();
	return(*this);
}
__device__ __host__ inline Transform& Transform::upscale(const Vector3 &sclDelta){
	scale += sclDelta;
	rebake();
	return(*this);
}





/** -------------------------------------------------------------------------- **/
/** Transform & detransform: **/
__device__ __host__ inline Vector3& operator>>=(Vector3 &v, const Transform &t){
	return(v(v*t.trans.x + t.position.x, v*t.trans.y + t.position.y, v*t.trans.z + t.position.z));
}
__device__ __host__ inline Vector3 operator>>(const Vector3 &v, const Transform &t){
	return(Vector3(v*t.trans.x + t.position.x, v*t.trans.y + t.position.y, v*t.trans.z + t.position.z));
}

__device__ __host__ inline Vector3& operator<<=(Vector3 &v, const Transform &t){
	Vector3 delta = v - t.position;
	return(v(delta*t.detrans.x, delta*t.detrans.y, delta*t.detrans.z));
}
__device__ __host__ inline Vector3 operator<<(const Vector3 &v, const Transform &t){
	Vector3 delta = v - t.position;
	return(Vector3(delta*t.detrans.x, delta*t.detrans.y, delta*t.detrans.z));
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
