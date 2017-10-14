#include"Vector3.h"

/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
// Macros:

/** ------------------------------------ **/
#define SqrMagnitudeOfThis (x*x + y*y + z*z)
#define MagnitudeOfThis sqrt(x*x + y*y + z*z)
#define SqrMagnitude(v) (v.x*v.x + v.y*v.y + v.z*v.z)
#define Magnitude(v) sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

/** ------------------------------------ **/
#define DotProductWith(v) (x*v.x + y*v.y + z*v.z)
#define DotProduct(v1, v2) (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z)

/** ------------------------------------ **/
#define CrossWith_X_Param(v) (y*v.z - v.y*z)
#define CrossWith_Y_Param(v) (v.x*z - x*v.z)
#define CrossWith_Z_Param(v) (x*v.y - v.x*y)
#define CrossProductWith(v) Vector3(CrossWith_X_Param(v), CrossWith_Y_Param(v), CrossWith_Z_Param(v))
#define CrossProduct_X_Param(v1, v2) (v1.y*v2.z - v2.y*v1.z)
#define CrossProduct_Y_Param(v1, v2) (v2.x*v1.z - v1.x*v2.z)
#define CrossProduct_Z_Param(v1, v2) (v1.x*v2.y - v2.x*v1.y)
#define CrossProduct(v1, v2) Vector3(CrossProduct_X_Param(v1, v2), CrossProduct_Y_Param(v1, v2), CrossProduct_Z_Param(v1, v2))

/** ------------------------------------ **/
#define AngleCosWith(v) DotProductWith(v) / sqrt(SqrMagnitudeOfThis*SqrMagnitude(v))
#define AngleCos(v1, v2) DotProduct(v1, v2) / sqrt(SqrMagnitude(v1)*SqrMagnitude(v2))

/** ------------------------------------ **/
#define NormalOn(v) (v*(DotProductWith(v) / SqrMagnitude(v)))

/** ------------------------------------ **/
#define DefCross register float crossX, crossY, crossZ;
#define AssignCrossWith(axis) crossX = CrossWith_X_Param(axis); crossY = CrossWith_Y_Param(axis); crossZ = CrossWith_Z_Param(axis)
#define AssignCrossOf(v1,v2) crossX = CrossProduct_X_Param(v1, v2); crossY = CrossProduct_Y_Param(v1, v2); crossZ = CrossProduct_Z_Param(v1, v2)
#define DefCrossWith(v) DefCross; AssignCrossWith(v)
#define DefCrossBetween(v1, v2) DefCross; AssignCrossOf(v1,v2)

/** ------------------------------------ **/
#define AssignAndReturnThis(a, b, c) x=a; y=b; z=c; return(*this)





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Construction: **/
// Does nothing
__dumb__ Vector3::Vector3(){}
// Simple constructor
__dumb__ Vector3::Vector3(float X, float Y, float Z){
	x = X;
	y = Y;
	z = Z;
}
// Copy constructor
__dumb__ Vector3::Vector3(const Vector3 &v){
	x = v.x;
	y = v.y;
	z = v.z;
}
// Reconstructor
__dumb__ Vector3& Vector3::operator()(float X, float Y, float Z){
	AssignAndReturnThis(X, Y, Z);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Operators: **/


/** ========================================================== **/
/*| + |*/
// Enables writing "+v"
__dumb__ Vector3 Vector3::operator+()const{
	return(*this);
}
// Sum of two vectors
__dumb__ Vector3 Vector3::operator+(const Vector3 &v)const{
	return(Vector3(x + v.x, y + v.y, z + v.z));
}
// Increment
__dumb__ Vector3& Vector3::operator+=(const Vector3 &v){
	x += v.x;
	y += v.y;
	z += v.z;
	return(*this);
}


/** ========================================================== **/
/*| - |*/
// Inversed vector
__dumb__ Vector3 Vector3::operator-()const{
	return(Vector3(-x, -y, -z));
}
// Subtraction of two vectors
__dumb__ Vector3 Vector3::operator-(const Vector3 &v)const{
	return(Vector3(x - v.x, y - v.y, z - v.z));
}
// Decrement
__dumb__ Vector3& Vector3::operator-=(const Vector3 &v){
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return(*this);
}


/** ========================================================== **/
/*| * & ^ |*/

/** ------------------------------------ **/
// Multiplied by floating point
__dumb__ Vector3 Vector3::operator*(const float f)const{
	return(Vector3(x*f, y*f, z*f));
}
// Inversed syntax for floating point multiplication
__dumb__ Vector3 operator*(const float f, const Vector3 &v){
	return(Vector3(v.x*f, v.y*f, v.z*f));
}
// Multilication by a floating point
__dumb__ Vector3& Vector3::operator*=(const float f){
	x *= f;
	y *= f;
	z *= f;
	return(*this);
}

/** ------------------------------------ **/
// Dot product
__dumb__ float Vector3::operator*(const Vector3 &v)const{
	return(DotProductWith(v));
}

/** ------------------------------------ **/
// Cross product
__dumb__ Vector3 Vector3::operator&(const Vector3 &v)const{
	return(CrossProductWith(v));
}
// Sets vector to it's cross product with the another one
__dumb__ Vector3& Vector3::operator&=(const Vector3 &v){
	AssignAndReturnThis(CrossWith_X_Param(v), CrossWith_Y_Param(v), CrossWith_Z_Param(v));
}

/** ------------------------------------ **/
// Upscaled (x*v.x, y*v.y, z*v.z)
__dumb__ Vector3 Vector3::operator^(const Vector3 &v)const{
	return(Vector3(x*v.x, y*v.y, z*v.z));
}
// Upscale
__dumb__ Vector3& Vector3::operator^=(const Vector3 &v){
	x *= v.x;
	y *= v.y;
	z *= v.z;
	return(*this);
}


/** ========================================================== **/
/*| / |*/

/** ------------------------------------ **/
// Divided by floating point
__dumb__ Vector3 Vector3::operator/(const float f)const{
	register float fi = 1 / f;
	return(Vector3(x*fi, y*fi, z*fi));
}
// Inversed syntax for floating point division(dumb and unnessessary, but anyway...)
__dumb__ Vector3 operator/(const float f, const Vector3 &v){
	return(Vector3(f / v.x, f / v.y, f / v.z));
}
// Division by floating point
__dumb__ Vector3& Vector3::operator/=(const float f){
	register float fi = 1 / f;
	x *= fi;
	y *= fi;
	z *= fi;
	return(*this);
}

/** ------------------------------------ **/
// Downscaled (x/v.x, y/v.y, z/v.y)
__dumb__ Vector3 Vector3::operator/(const Vector3 &v)const{
	return(Vector3(x / v.x, y / v.y, z / v.z));
}
// Downscale
__dumb__ Vector3& Vector3::operator/=(const Vector3 &v){
	x /= v.x;
	y /= v.y;
	z /= v.z;
	return(*this);
}


/** ========================================================== **/
/*| == |*/
// Compare (equals)
__dumb__ bool Vector3::operator==(const Vector3 &v)const{
	return((x == v.x) && (y == v.y) && (z == v.z));
}
// Compare (not equals)
__dumb__ bool Vector3::operator!=(const Vector3 &v)const{
	return((x != v.x) || (y != v.y) || (z != v.z));
}
// Compare (distance between is low enough)
__dumb__ bool Vector3::isNearTo(const Vector3 &v, const float maxDistance)const{
	register float dx = x - v.x;
	register float dy = y - v.y;
	register float dz = z - v.z;
	return((dx*dx + dy*dy + dz*dz) <= (maxDistance*maxDistance));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Functions: **/


/** ========================================================== **/
/*| Magnitude |*/

/** ------------------------------------ **/
// Square of the vector's length
__dumb__ float Vector3::sqrMagnitude()const{
	return(SqrMagnitudeOfThis);
}
// Vector's length
__dumb__ float Vector3::magnitude()const{
	return(MagnitudeOfThis);
}

/** ------------------------------------ **/
// Distance to the other vertex
__dumb__ float Vector3::distanceTo(const Vector3 &v)const{
	register float dx = x - v.x;
	register float dy = y - v.y;
	register float dz = z - v.z;
	return(sqrt(dx*dx + dy*dy + dz*dz));
}
// Distance between vertices
__dumb__ float Vector3::distance(const Vector3 &v1, const Vector3 &v2){
	register float dx = v1.x - v2.x;
	register float dy = v1.y - v2.y;
	register float dz = v1.z - v2.z;
	return(sqrt(dx*dx + dy*dy + dz*dz));
}

/** ------------------------------------ **/
// Unit vector with the same direction
__dumb__ Vector3 Vector3::normalized()const{
	register float inversedMagnitude = 1 / MagnitudeOfThis;
	return(Vector3(x*inversedMagnitude, y*inversedMagnitude, z*inversedMagnitude));
}
// Makes this vector unit, without changing direction
__dumb__ Vector3& Vector3::normalize(){
	register float inversedMagnitude = 1 / MagnitudeOfThis;
	x *= inversedMagnitude;
	y *= inversedMagnitude;
	z *= inversedMagnitude;
	return((*this));
}


/** ========================================================== **/
/*| Angle |*/

/** ------------------------------------ **/
// Cosine of angle between vectors
__dumb__ float Vector3::angleCos(const Vector3 &v1, const Vector3 &v2){
	return(AngleCos(v1, v2));
}
// Cosine of angle between unit vectors
__dumb__ float Vector3::angleCosUnitVectors(const Vector3 &v1, const Vector3 &v2){
	return(DotProduct(v1, v2));
}
// Cosine of angle between this and the another vector
__dumb__ float Vector3::angleCos(const Vector3 &v)const{
	return(AngleCosWith(v));
}
// Cosine of angle between this and the another unit vector
__dumb__ float Vector3::angleCosUnitVectors(const Vector3 &v)const{
	return(DotProductWith(v));
}

/** ------------------------------------ **/
// Sine of angle between vectors
__dumb__ float Vector3::angleSin(const Vector3 &v1, const Vector3 &v2){
	DefCrossBetween(v1, v2);
	return(sqrt((crossX*crossX + crossY*crossY + crossZ*crossZ) / (SqrMagnitude(v1)*SqrMagnitude(v2))));
}
// Sine of angle between unit vectors
__dumb__ float Vector3::angleSinUnitVectors(const Vector3 &v1, const Vector3 &v2){
	DefCrossBetween(v1, v2);
	return(sqrt(crossX*crossX + crossY*crossY + crossZ*crossZ));
}
// Sine of angle between this and the another vector
__dumb__ float Vector3::angleSin(const Vector3 &v)const{
	DefCrossWith(v);
	return(sqrt((crossX*crossX + crossY*crossY + crossZ*crossZ) / (SqrMagnitudeOfThis*SqrMagnitude(v))));
}
// Sine of angle between this and the another unit vector
__dumb__ float Vector3::angleSinUnitVectors(const Vector3 &v)const{
	DefCrossWith(v);
	return(sqrt(crossX*crossX + crossY*crossY + crossZ*crossZ));
}

/** ------------------------------------ **/
// Angle between two vectors in radians
__dumb__ float Vector3::radianAngle(const Vector3 &v1, const Vector3 &v2){
	return(acos(AngleCos(v1, v2)));
}
// Angle between two unit vectors in radians
__dumb__ float Vector3::radianAngleUnitVectors(const Vector3 &v1, const Vector3 &v2){
	return(acos(DotProduct(v1, v2)));
}
// Angle between this and the another vector in radians
__dumb__ float Vector3::radianAngle(const Vector3 &v)const{
	return(acos(AngleCosWith(v)));
}
// Angle between this and the another unit vector in radians
__dumb__ float Vector3::radianAngleUnitVectors(const Vector3 &v)const{
	return(acos(DotProductWith(v)));
}

/** ------------------------------------ **/
// Angle between two vectors in degrees
__dumb__ float Vector3::angle(const Vector3 &v1, const Vector3 &v2){
	return(acos(AngleCos(v1, v2))*RADIAN);
}
// Angle between two unit vectors in degrees
__dumb__ float Vector3::angleUnitVectors(const Vector3 &v1, const Vector3 &v2){
	return(acos(DotProduct(v1, v2))*RADIAN);
}
// Angle between this and the another vector in degrees
__dumb__ float Vector3::angle(const Vector3 &v)const{
	return(acos(AngleCosWith(v))*RADIAN);
}
// Angle between this and the another unit vector in degrees
__dumb__ float Vector3::angleUnitVectors(const Vector3 &v)const{
	return(acos(DotProductWith(v))*RADIAN);
}


/** ========================================================== **/
/*| Normal |*/

/** ------------------------------------ **/
// Normal(projection) on another vector
__dumb__ Vector3 Vector3::normalOn(const Vector3 &v)const{
	return(NormalOn(v));
}
// Normal(projection) on an unit vector
__dumb__ Vector3 Vector3::normalOnUnitVector(const Vector3 &v)const{
	return(v*DotProductWith(v));
}
// Projects on another vector
__dumb__ Vector3& Vector3::normalizeOn(const Vector3 &v){
	return((*this) = NormalOn(v));
}
// Projects on an unit vector
__dumb__ Vector3& Vector3::normalizeOnUnitVector(const Vector3 &v){
	return((*this) = v*DotProductWith(v));
}

/** ------------------------------------ **/
// Magnitude of the normal on the another vector
__dumb__ float Vector3::normalLengthOn(const Vector3 &v)const{
	return(DotProductWith(v) / Magnitude(v));
}
// Magnitude of the normal on the unit vector
__dumb__ float Vector3::normalLengthOnUnitVector(const Vector3 &v)const{
	return(DotProductWith(v));
}

/** ------------------------------------ **/
// Noraml of the plain, defined by two vectors
__dumb__ Vector3 Vector3::plainNormal(const Vector3 &v1, const Vector3 &v2){
	DefCrossBetween(v1, v2);
	register float inevrsedMagnitude = 1 / sqrt(crossX*crossX + crossY*crossY + crossZ*crossZ);
	return(Vector3(crossX*inevrsedMagnitude, crossY*inevrsedMagnitude, crossZ*inevrsedMagnitude));
}
// Noraml of the plain, defined by two unit vectors
__dumb__ Vector3 Vector3::plainNormalForUnitVectors(const Vector3 &v1, const Vector3 &v2){
	return(CrossProduct(v1, v2));
}
// Noraml of the plain, defined by three vertices in space
__dumb__ Vector3 Vector3::plainNormal(const Vector3 &v1, const Vector3 &v2, const Vector3 &v3){
	register float v1x = v2.x - v1.x;
	register float v1y = v2.y - v1.y;
	register float v1z = v2.z - v1.z;
	register float v2x = v3.x - v1.x;
	register float v2y = v3.y - v1.y;
	register float v2z = v3.z - v1.z;
	register float crossX = (v2y*v1z - v1y*v2z);
	register float crossY = (v1x*v2z - v2x*v1z);
	register float crossZ = (v2x*v1y - v1x*v2y);
	register float inevrsedMagnitude = 1 / sqrt(crossX*crossX + crossY*crossY + crossZ*crossZ);
	return(Vector3(crossX*inevrsedMagnitude, crossY*inevrsedMagnitude, crossZ*inevrsedMagnitude));
}

/** ------------------------------------ **/
// Reflection
__dumb__ Vector3 Vector3::reflection(const Vector3 &normal)const {
	register float mul = (2.0f * DotProductWith(normal) / SqrMagnitude(normal));
	return Vector3(x - (normal.x * mul), y - (normal.y * mul), z - (normal.z * mul));
}
__dumb__ Vector3& Vector3::reflect(const Vector3 &normal) {
	register float mul = (2.0f * DotProductWith(normal) / SqrMagnitude(normal));
	x -= (normal.x * mul);
	y -= (normal.y * mul);
	z -= (normal.z * mul);
	return(*this);
}
__dumb__ Vector3 Vector3::reflectionOnUnitVector(const Vector3 &normal)const {
	register float mul = (2.0f * DotProductWith(normal));
	return Vector3(x - (normal.x * mul), y - (normal.y * mul), z - (normal.z * mul));
}
__dumb__ Vector3& Vector3::reflectOnUnitVector(const Vector3 &normal) {
	register float mul = (2.0f * DotProductWith(normal));
	x -= (normal.x * mul);
	y -= (normal.y * mul);
	z -= (normal.z * mul);
	return (*this);
}


/** ========================================================== **/
/*| Rotation |*/

/** ------------------------------------ **/
#define DefNorm register float normalMultiplier = DotProductWith(axis) / SqrMagnitude(axis); register float normX = axis.x*normalMultiplier; register float normY = axis.y*normalMultiplier; register float normZ = axis.z*normalMultiplier
#define DefDelta register float dx = x - normX; register float dy = y - normY; register float dz = z - normZ
#define DefMultipliers register float cosA = cos(angle); register float sinMultiplier = sin(angle)*sqrt((dx*dx + dy*dy + dz*dz) / (crossX*crossX + crossY*crossY + crossZ*crossZ))
#define DefRotationData DefNorm; DefDelta; DefCross; AssignCrossWith(axis); DefMultipliers
#define RotationResultParams normX + dx*cosA + crossX*sinMultiplier, normY + dy*cosA + crossY*sinMultiplier, normZ + dz*cosA + crossZ*sinMultiplier

/** ------------------------------------ **/
#define DefSinCos register float sinA = sin(angle); register float cosA = cos(angle);
#define DefSinCosAndCross DefSinCos; DefCross; AssignCrossWith(axis)
#define RotationAxisAgainstAxisParams x*cosA + crossX*sinA, y*cosA + crossY*sinA, z*cosA + crossZ*sinA

/** ------------------------------------ **/
#define RotatedAxisParams(v) v.x*cosA - crossX*sinA, v.y*cosA - crossY*sinA, v.z*cosA - crossZ*sinA
#define TwoAxisRotationRoutine DefSinCos; DefCross; AssignCrossWith(v1); v1(RotatedAxisParams(v1)); AssignCrossWith(v2); v2(RotatedAxisParams(v2))

/** ------------------------------------ **/
// This vector, rotated against given vector by given radian angle
__dumb__ Vector3 Vector3::rotated_Radian(const Vector3 &axis, float angle)const{
	DefRotationData;
	return(Vector3(RotationResultParams));
}
// This vector, rotated against given axis by given radian angle, if both vectors are unit and perpendicular to each other
__dumb__ Vector3 Vector3::rotated_AxisAgainstAxisRadian(const Vector3 &axis, float angle)const{
	DefSinCosAndCross;
	return(Vector3(RotationAxisAgainstAxisParams));
}
// Rotates vector against given vector by given radian angle
__dumb__ Vector3& Vector3::rotate_Radian(const Vector3 &axis, float angle){
	DefRotationData;
	return((*this)(RotationResultParams));
}
// Rotates vector against given vector by given radian angle, if both vectors are unit and perpendicular to each other
__dumb__ Vector3& Vector3::rotate_AxisAgainstAxisRadian(const Vector3 &axis, float angle){
	DefSinCosAndCross;
	return((*this)(RotationAxisAgainstAxisParams));
}

/** ------------------------------------ **/
// This vector, rotated against given vector by given angle in degrees
__dumb__ Vector3 Vector3::rotated(const Vector3 &axis, float angle)const{
	angle /= RADIAN;
	DefRotationData;
	return(Vector3(RotationResultParams));
}
// This vector, rotated against given axis by given angle in degrees, if both vectors are unit and perpendicular to each other
__dumb__ Vector3 Vector3::rotated_AxisAgainstAxis(const Vector3 &axis, float angle)const{
	angle /= RADIAN;
	DefSinCosAndCross;
	return(Vector3(RotationAxisAgainstAxisParams));
}
// Rotates vector against given vector by given angle in degrees
__dumb__ Vector3& Vector3::rotate(const Vector3 &axis, float angle){
	angle /= RADIAN;
	DefRotationData;
	return((*this)(RotationResultParams));
}
// Rotates vector against given vector by given angle in degrees, if both vectors are unit and perpendicular to each other
__dumb__ Vector3& Vector3::rotate_AxisAgainstAxis(const Vector3 &axis, float angle){
	angle /= RADIAN;
	DefSinCosAndCross;
	return((*this)(RotationAxisAgainstAxisParams));
}

/** ------------------------------------ **/
// Rotates two vectors by given radian angle against itself, if all three vectors are unit and perpendicular to each other
__dumb__ void Vector3::rotateTwoAxisAgainstThisRadian(Vector3 &v1, Vector3 &v2, float angle)const{
	TwoAxisRotationRoutine;
}
// Rotates two vectors by given angle in degrees against itself, if all three vectors are unit and perpendicular to each other
__dumb__ void Vector3::rotateTwoAxisAgainstThis(Vector3 &v1, Vector3 &v2, float angle)const{
	angle /= RADIAN;
	TwoAxisRotationRoutine;
}

/** ------------------------------------ **/
#undef DefNorm
#undef DefDelta
#undef DefMultipliers
#undef DefRotationData
#undef RotationResultParams

/** ------------------------------------ **/
#undef DefSinCos
#undef DefSinCosAndCross
#undef RotationAxisAgainstAxisParams

/** ------------------------------------ **/
#undef RotatedAxisParams
#undef TwoAxisRotationRoutine





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Constants: **/

/** ========================================================== **/
/*| Variables |*/

/** ------------------------------------ **/
#define VECTOR3_ZERO { 0, 0, 0 }
const Vector3 VECTOR3_ZERO_H = VECTOR3_ZERO;
__device__ __constant__ const float3 VECTOR3_ZERO_D = VECTOR3_ZERO;
#define VECTOR3_ONE { 1, 1, 1 }
const Vector3 VECTOR3_ONE_H = VECTOR3_ONE;
__device__ __constant__ const float3 VECTOR3_ONE_D = VECTOR3_ONE;
#define VECTOR3_UP { 0, 1, 0 }
const Vector3 VECTOR3_UP_H = VECTOR3_UP;
__device__ __constant__ const float3 VECTOR3_UP_D = VECTOR3_UP;
#define VECTOR3_DOWN { 0, -1, 0 }
const Vector3 VECTOR3_DOWN_H = VECTOR3_DOWN;
__device__ __constant__ const float3 VECTOR3_DOWN_D = VECTOR3_DOWN;
#define VECTOR3_FRONT { 0, 0, 1 }
const Vector3 VECTOR3_FRONT_H = VECTOR3_FRONT;
__device__ __constant__ const float3 VECTOR3_FRONT_D = VECTOR3_FRONT;
#define VECTOR3_BACK { 0, 0, -1 }
const Vector3 VECTOR3_BACK_H = VECTOR3_BACK;
__device__ __constant__ const float3 VECTOR3_BACK_D = VECTOR3_BACK;
#define VECTOR3_RIGHT { 1, 0, 0 }
const Vector3 VECTOR3_RIGHT_H = VECTOR3_RIGHT;
__device__ __constant__ const float3 VECTOR3_RIGHT_D = VECTOR3_RIGHT;
#define VECTOR3_LEFT { -1, 0, 0 }
const Vector3 VECTOR3_LEFT_H = VECTOR3_LEFT;
__device__ __constant__ const float3 VECTOR3_LEFT_D = VECTOR3_LEFT;

/** ------------------------------------ **/
#define VECTOR3_X_AXIS { 1, 0, 0 }
const Vector3 VECTOR3_X_AXIS_H = VECTOR3_X_AXIS;
__device__ __constant__ const float3 VECTOR3_X_AXIS_D = VECTOR3_X_AXIS;
#define VECTOR3_Y_AXIS { 0, 1, 0 }
const Vector3 VECTOR3_Y_AXIS_H = VECTOR3_Y_AXIS;
__device__ __constant__ const float3 VECTOR3_Y_AXIS_D = VECTOR3_Y_AXIS;
#define VECTOR3_Z_AXIS { 0, 0, 1 }
const Vector3 VECTOR3_Z_AXIS_H = VECTOR3_Z_AXIS;
__device__ __constant__ const float3 VECTOR3_Z_AXIS_D = VECTOR3_Z_AXIS;
#undef VECTOR3_EQIVALENT_FLOAT_ARRAY_SIZE

/** ========================================================== **/
/*| GPU initialisation |*/
// Initializes constants on GPU
inline bool Vector3::initConstants(){
	if (cudaMemcpyToSymbol(&VECTOR3_ZERO_D, &VECTOR3_ZERO_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(&VECTOR3_ONE_D, &VECTOR3_ONE_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(&VECTOR3_UP_D, &VECTOR3_UP_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(&VECTOR3_DOWN_D, &VECTOR3_DOWN_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(&VECTOR3_FRONT_D, &VECTOR3_FRONT_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(&VECTOR3_BACK_D, &VECTOR3_BACK_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(&VECTOR3_RIGHT_D, &VECTOR3_RIGHT_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(&VECTOR3_LEFT_D, &VECTOR3_LEFT_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);

	if (cudaMemcpyToSymbol(&VECTOR3_X_AXIS_D, &VECTOR3_X_AXIS_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(&VECTOR3_Y_AXIS_D, &VECTOR3_Y_AXIS_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(&VECTOR3_Z_AXIS_D, &VECTOR3_Z_AXIS_H, sizeof(Vector3), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);

	return(true);
}

/** ========================================================== **/
/*| Access |*/

/** ------------------------------------ **/
// Vector3(0, 0, 0); Has visible constants VECTOR3_ZERO_H and *VECTOR3_ZERO_D for the host and the device, respectively, as well as macro VECTOR3_ZERO.
__dumb__ const Vector3& Vector3::zero(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_ZERO_D)));
#else
	return VECTOR3_ZERO_H;
#endif
}
// Vector3(1, 1, 1); Has visible constants VECTOR3_ONE_H and *VECTOR3_ONE_D for the host and the device, respectively, as well as macro VECTOR3_ONE.
__dumb__ const Vector3& Vector3::one(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_ONE_D)));
#else
	return VECTOR3_ONE_H;
#endif
}
// Vector3(0, 1, 0); Has visible constants VECTOR3_UP_H and *VECTOR3_UP_D for the host and the device, respectively, as well as macro VECTOR3_UP.
__dumb__ const Vector3& Vector3::up(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_UP_D)));
#else
	return VECTOR3_UP_H;
#endif
}
// Vector3(0, -1, 0); Has visible constants VECTOR3_DOWN_H and *VECTOR3_DOWN_D for the host and the device, respectively, as well as macro VECTOR3_DOWN.
__dumb__ const Vector3& Vector3::down(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_DOWN_D)));
#else
	return VECTOR3_DOWN_H;
#endif
}
// Vector3(0, 0, 1); Has visible constants VECTOR3_FRONT_H and *VECTOR3_FRONT_D for the host and the device, respectively, as well as macro VECTOR3_FRONT.
__dumb__ const Vector3& Vector3::front(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_FRONT_D)));
#else
	return VECTOR3_FRONT_H;
#endif
}
// Vector3(0, 0, -1); Has visible constants VECTOR3_BACK_H and *VECTOR3_BACK_D for the host and the device, respectively, as well as macro VECTOR3_BACK.
__dumb__ const Vector3& Vector3::back(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_BACK_D)));
#else
	return VECTOR3_BACK_H;
#endif
}
// Vector3(1, 0, 0); Has visible constants VECTOR3_RIGHT_H and *VECTOR3_RIGHT_D for the host and the device, respectively, as well as macro VECTOR3_RIGHT.
__dumb__ const Vector3& Vector3::right(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_RIGHT_D)));
#else
	return VECTOR3_RIGHT_H;
#endif
}
// Vector3(-1, 0, 0); Has visible constants VECTOR3_LEFT_H and *VECTOR3_LEFT_D for the host and the device, respectively, as well as macro VECTOR3_LEFT.
__dumb__ const Vector3& Vector3::left(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_LEFT_D)));
#else
	return VECTOR3_LEFT_H;
#endif
}

/** ------------------------------------ **/
// Vector3(1, 0, 0); Has visible constants VECTOR3_X_AXIS_H and *VECTOR3_X_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_X_AXIS.
__dumb__ const Vector3& Vector3::Xaxis(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_X_AXIS_D)));
#else
	return VECTOR3_X_AXIS_H;
#endif
}
// Vector3(0, 1, 0); Has visible constants VECTOR3_Y_AXIS_H and *VECTOR3_Y_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_Y_AXIS.
__dumb__ const Vector3& Vector3::Yaxis(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_Y_AXIS_D)));
#else
	return VECTOR3_Y_AXIS_H;
#endif
}
// Vector3(0, 0, 1); Has visible constants VECTOR3_Z_AXIS_H and *VECTOR3_Z_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_Z_AXIS.
__dumb__ const Vector3& Vector3::Zaxis(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_Z_AXIS_D)));
#else
	return VECTOR3_Z_AXIS_H;
#endif
}
// Vector3(1, 0, 0); Has visible constants VECTOR3_X_AXIS_H and *VECTOR3_X_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_X_AXIS.
__dumb__ const Vector3& Vector3::i(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_X_AXIS_D)));
#else
	return VECTOR3_X_AXIS_H;
#endif
}
// Vector3(0, 1, 0); Has visible constants VECTOR3_Y_AXIS_H and *VECTOR3_Y_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_Y_AXIS.
__dumb__ const Vector3& Vector3::j(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_Y_AXIS_D)));
#else
	return VECTOR3_Y_AXIS_H;
#endif
}
// Vector3(0, 0, 1); Has visible constants VECTOR3_Z_AXIS_H and *VECTOR3_Z_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_Z_AXIS.
__dumb__ const Vector3& Vector3::k(){
#ifdef __CUDA_ARCH__
	return (*((Vector3*)(&VECTOR3_Z_AXIS_D)));
#else
	return VECTOR3_Z_AXIS_H;
#endif
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Stream operators: **/
// Stream operator for input.
inline std::istream& operator>>(std::istream &stream, Vector3 &v){
	return(stream >> v.x >> v.y >> v.z);
}
// Stream operator for output.
inline std::ostream& operator<<(std::ostream &stream, const Vector3 &v){
	return(stream << "(" << v.x << ", " << v.y << ", " << v.z << ")");
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Macro undefs: **/

/** ------------------------------------ **/
#undef SqrMagnitudeOfThis
#undef MagnitudeOfThis
#undef SqrMagnitude
#undef Magnitude

/** ------------------------------------ **/
#undef DotProductWith
#undef DotProduct

/** ------------------------------------ **/
#undef CrossWith_X_Param
#undef CrossWith_Y_Param
#undef CrossWith_Z_Param
#undef CrossProductWith
#undef CrossProduct_X_Param
#undef CrossProduct_Y_Param
#undef CrossProduct_Z_Param
#undef CrossProduct

/** ------------------------------------ **/
#undef AngleCosWith
#undef AngleCos

/** ------------------------------------ **/
#undef NormalOn

/** ------------------------------------ **/
#undef DefCross
#undef AssignCrossWith
#undef AssignCrossOf
#undef DefCrossWith
#undef DefCrossBetween

/** ------------------------------------ **/
#undef AssignAndReturnThis
