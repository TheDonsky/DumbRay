#include"Vector2.h"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
// Macros:

/** ------------------------------------ **/
#define SqrMagnitudeOfThisVector2 (x*x + y*y)
#define MagnitudeOfThisVector2 sqrt(x*x + y*y)
#define SqrMagnitudeVector2(v) (v.x*v.x + v.y*v.y)
#define MagnitudeVector2(v) sqrt(v.x*v.x + v.y*v.y)

/** ------------------------------------ **/
#define DotProductWithVector2(v) (x*v.x + y*v.y)
#define DotProductVector2(v1, v2) (v1.x*v2.x + v1.y*v2.y)

/** ------------------------------------ **/
#define CrossWith_Z_Vector2(v) (x*v.y - v.x*y)
#define CrossProduct_Z_Vector2(v1, v2) (v1.x*v2.y - v2.x*v1.y)

/** ------------------------------------ **/
#define AngleCosWithVector2(v) DotProductWithVector2(v) / sqrt(SqrMagnitudeOfThisVector2*SqrMagnitudeVector2(v))
#define AngleCosVector2(v1, v2) DotProductVector2(v1, v2) / sqrt(SqrMagnitudeVector2(v1)*SqrMagnitudeVector2(v2))

/** ------------------------------------ **/
#define NormalOnVector2(v) (v*(DotProductWithVector2(v) / SqrMagnitudeVector2(v)))

/** ------------------------------------ **/
#define DefCrossVector2 register float crossZ;
#define AssignCrossWithVector2(axis) crossZ = CrossWith_Z_Vector2(axis)
#define AssignCrossOfVector2(v1,v2) crossZ = CrossProduct_Z_Vector2(v1, v2)
#define DefCrossWithVector2(v) DefCrossVector2; AssignCrossWithVector2(v)
#define DefCrossBetweenVector2(v1, v2) DefCrossVector2; AssignCrossOfVector2(v1,v2)

/** ------------------------------------ **/
#define AssignAndReturnThisVector2(a, b) x=a; y=b; return(*this)





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Construction: **/
// Does nothing
__dumb__ Vector2::Vector2() {}
// Simple constructor
__dumb__ Vector2::Vector2(float X, float Y) {
	x = X;
	y = Y;
}
// Copy constructor
__dumb__ Vector2::Vector2(const Vector2 &v) {
	x = v.x;
	y = v.y;
}
// Cast constructor
__dumb__ Vector2::Vector2(const Vector3 &v) {
	x = v.x;
	y = v.y;
}
// Cast to Vector3
__dumb__ Vector2::operator Vector3()const {
	return Vector3(x, y, 0);
}
// Reconstructor
__dumb__ Vector2& Vector2::operator()(float X, float Y) {
	AssignAndReturnThisVector2(X, Y);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Operators: **/


/** ========================================================== **/
/*| + |*/
// Enables writing "+v"
__dumb__ Vector2 Vector2::operator+()const {
	return(*this);
}
// Sum of two vectors
__dumb__ Vector2 Vector2::operator+(const Vector2 &v)const {
	return(Vector2(x + v.x, y + v.y));
}
// Increment
__dumb__ Vector2& Vector2::operator+=(const Vector2 &v) {
	x += v.x;
	y += v.y;
	return(*this);
}


/** ========================================================== **/
/*| - |*/
// Inversed vector
__dumb__ Vector2 Vector2::operator-()const {
	return(Vector2(-x, -y));
}
// Subtraction of two vectors
__dumb__ Vector2 Vector2::operator-(const Vector2 &v)const {
	return(Vector2(x - v.x, y - v.y));
}
// Decrement
__dumb__ Vector2& Vector2::operator-=(const Vector2 &v) {
	x -= v.x;
	y -= v.y;
	return(*this);
}


/** ========================================================== **/
/*| * & ^ |*/

/** ------------------------------------ **/
// Multiplied by floating point
__dumb__ Vector2 Vector2::operator*(const float f)const {
	return(Vector2(x*f, y*f));
}
// Inversed syntax for floating point multiplication
__dumb__ Vector2 operator*(const float f, const Vector2 &v) {
	return(Vector2(v.x*f, v.y*f));
}
// Multilication by a floating point
__dumb__ Vector2& Vector2::operator*=(const float f) {
	x *= f;
	y *= f;
	return(*this);
}

/** ------------------------------------ **/
// Dot product
__dumb__ float Vector2::operator*(const Vector2 &v)const {
	return(DotProductWithVector2(v));
}

/** ------------------------------------ **/
// Z coordinate of the cross product
__dumb__ float Vector2::operator&(const Vector2 &v)const {
	return(CrossWith_Z_Vector2(v));
}

/** ------------------------------------ **/
// Upscaled (x*v.x, y*v.y)
__dumb__ Vector2 Vector2::operator^(const Vector2 &v)const {
	return(Vector2(x*v.x, y*v.y));
}
// Upscale
__dumb__ Vector2& Vector2::operator^=(const Vector2 &v) {
	x *= v.x;
	y *= v.y;
	return(*this);
}


/** ========================================================== **/
/*| / |*/

/** ------------------------------------ **/
// Divided by floating point
__dumb__ Vector2 Vector2::operator/(const float f)const {
	register float fi = 1 / f;
	return(Vector2(x*fi, y*fi));
}
// Inversed syntax for floating point division(dumb and unnessessary, but anyway...)
__dumb__ Vector2 operator/(const float f, const Vector2 &v) {
	return(Vector2(f / v.x, f / v.y));
}
// Division by floating point
__dumb__ Vector2& Vector2::operator/=(const float f) {
	register float fi = 1 / f;
	x *= fi;
	y *= fi;
	return(*this);
}

/** ------------------------------------ **/
// Downscaled (x/v.x, y/v.y)
__dumb__ Vector2 Vector2::operator/(const Vector2 &v)const {
	return(Vector2(x / v.x, y / v.y));
}
// Downscale
__dumb__ Vector2& Vector2::operator/=(const Vector2 &v) {
	x /= v.x;
	y /= v.y;
	return(*this);
}


/** ========================================================== **/
/*| == |*/
// Compare (equals)
__dumb__ bool Vector2::operator==(const Vector2 &v)const {
	return((x == v.x) && (y == v.y));
}
// Compare (not equals)
__dumb__ bool Vector2::operator!=(const Vector2 &v)const {
	return((x != v.x) || (y != v.y));
}
// Compare (distance between is low enough)
__dumb__ bool Vector2::isNearTo(const Vector2 &v, const float maxDistance)const {
	register float dx = x - v.x;
	register float dy = y - v.y;
	return((dx*dx + dy*dy) <= (maxDistance*maxDistance));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Functions: **/


/** ========================================================== **/
/*| Magnitude |*/

/** ------------------------------------ **/
// Square of the vector's length
__dumb__ float Vector2::sqrMagnitude()const {
	return(SqrMagnitudeOfThisVector2);
}
// Vector's length
__dumb__ float Vector2::magnitude()const {
	return(MagnitudeOfThisVector2);
}

/** ------------------------------------ **/
// Distance to the other vertex
__dumb__ float Vector2::distanceTo(const Vector2 &v)const {
	register float dx = x - v.x;
	register float dy = y - v.y;
	return(sqrt(dx*dx + dy*dy));
}
// Distance between vertices
__dumb__ float Vector2::distance(const Vector2 &v1, const Vector2 &v2) {
	register float dx = v1.x - v2.x;
	register float dy = v1.y - v2.y;
	return(sqrt(dx*dx + dy*dy));
}

/** ------------------------------------ **/
// Unit vector with the same direction
__dumb__ Vector2 Vector2::normalized()const {
	register float inversedMagnitude = 1.0f / MagnitudeOfThisVector2;
	return(Vector2(x*inversedMagnitude, y*inversedMagnitude));
}
// Makes this vector unit, without changing direction
__dumb__ Vector2& Vector2::normalize() {
	register float inversedMagnitude = 1.0f / MagnitudeOfThisVector2;
	x *= inversedMagnitude;
	y *= inversedMagnitude;
	return((*this));
}


/** ========================================================== **/
/*| Angle |*/

/** ------------------------------------ **/
// Cosine of angle between vectors
__dumb__ float Vector2::angleCos(const Vector2 &v1, const Vector2 &v2) {
	return(AngleCosVector2(v1, v2));
}
// Cosine of angle between unit vectors
__dumb__ float Vector2::angleCosUnitVectors(const Vector2 &v1, const Vector2 &v2) {
	return(DotProductVector2(v1, v2));
}
// Cosine of angle between this and the another vector
__dumb__ float Vector2::angleCos(const Vector2 &v)const {
	return(AngleCosWithVector2(v));
}
// Cosine of angle between this and the another unit vector
__dumb__ float Vector2::angleCosUnitVectors(const Vector2 &v)const {
	return(DotProductWithVector2(v));
}

/** ------------------------------------ **/
// Sine of angle between vectors
__dumb__ float Vector2::angleSin(const Vector2 &v1, const Vector2 &v2) {
	return(CrossProduct_Z_Vector2(v1, v2) / sqrt(SqrMagnitudeVector2(v1)*SqrMagnitudeVector2(v2)));
}
// Sine of angle between unit vectors
__dumb__ float Vector2::angleSinUnitVectors(const Vector2 &v1, const Vector2 &v2) {
	return(CrossProduct_Z_Vector2(v1, v2));
}
// Sine of angle between this and the another vector
__dumb__ float Vector2::angleSin(const Vector2 &v)const {
	return(CrossWith_Z_Vector2(v) / sqrt(SqrMagnitudeOfThisVector2*SqrMagnitudeVector2(v)));
}
// Sine of angle between this and the another unit vector
__dumb__ float Vector2::angleSinUnitVectors(const Vector2 &v)const {
	return(CrossWith_Z_Vector2(v));
}

/** ------------------------------------ **/
// Angle between two vectors in radians
__dumb__ float Vector2::radianAngle(const Vector2 &v1, const Vector2 &v2) {
	return(acos(AngleCosVector2(v1, v2)));
}
// Angle between two unit vectors in radians
__dumb__ float Vector2::radianAngleUnitVectors(const Vector2 &v1, const Vector2 &v2) {
	return(acos(DotProductVector2(v1, v2)));
}
// Angle between this and the another vector in radians
__dumb__ float Vector2::radianAngle(const Vector2 &v)const {
	return(acos(AngleCosWithVector2(v)));
}
// Angle between this and the another unit vector in radians
__dumb__ float Vector2::radianAngleUnitVectors(const Vector2 &v)const {
	return(acos(DotProductWithVector2(v)));
}

/** ------------------------------------ **/
// Angle between two vectors in degrees
__dumb__ float Vector2::angle(const Vector2 &v1, const Vector2 &v2) {
	return(acos(AngleCosVector2(v1, v2))*RADIAN);
}
// Angle between two unit vectors in degrees
__dumb__ float Vector2::angleUnitVectors(const Vector2 &v1, const Vector2 &v2) {
	return(acos(DotProductVector2(v1, v2))*RADIAN);
}
// Angle between this and the another vector in degrees
__dumb__ float Vector2::angle(const Vector2 &v)const {
	return(acos(AngleCosWithVector2(v))*RADIAN);
}
// Angle between this and the another unit vector in degrees
__dumb__ float Vector2::angleUnitVectors(const Vector2 &v)const {
	return(acos(DotProductWithVector2(v))*RADIAN);
}


/** ========================================================== **/
/*| Normal |*/

/** ------------------------------------ **/
// Normal(projection) on another vector
__dumb__ Vector2 Vector2::normalOn(const Vector2 &v)const {
	return(NormalOnVector2(v));
}
// Normal(projection) on an unit vector
__dumb__ Vector2 Vector2::normalOnUnitVector(const Vector2 &v)const {
	return(v*DotProductWithVector2(v));
}
// Projects on another vector
__dumb__ Vector2& Vector2::normalizeOn(const Vector2 &v) {
	return((*this) = NormalOnVector2(v));
}
// Projects on an unit vector
__dumb__ Vector2& Vector2::normalizeOnUnitVector(const Vector2 &v) {
	return((*this) = v*DotProductWithVector2(v));
}

/** ------------------------------------ **/
// Magnitude of the normal on the another vector
__dumb__ float Vector2::normalLengthOn(const Vector2 &v)const {
	return(DotProductWithVector2(v) / MagnitudeVector2(v));
}
// Magnitude of the normal on the unit vector
__dumb__ float Vector2::normalLengthOnUnitVector(const Vector2 &v)const {
	return(DotProductWithVector2(v));
}


/** ========================================================== **/
/*| Rotation |*/

/** ------------------------------------ **/
// This vector, rotated by given radian angle
__dumb__ Vector2 Vector2::rotated_Radian(float angle)const {
	register Vector2 normal(-y, x);
	return (((*this) * cos(angle)) + (normal * sin(angle)));
}
// Rotates vector by given radian angle
__dumb__ Vector2& Vector2::rotate_Radian(float angle) {
	return ((*this) = rotated_Radian(angle));
}

/** ------------------------------------ **/
// This vector, rotated by given angle in degrees
__dumb__ Vector2 Vector2::rotated(float angle)const {
	return rotated_Radian(angle / RADIAN);
}
// Rotates vector by given angle in degrees
__dumb__ Vector2& Vector2::rotate(float angle) {
	return ((*this) = rotated_Radian(angle / RADIAN));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Constants: **/

/** ========================================================== **/
/*| Variables |*/

/** ------------------------------------ **/
#define VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE ((sizeof(Vector2) + sizeof(float) - 1) / sizeof(float))
#define VECTOR2_ZERO { 0, 0 }
const Vector2 VECTOR2_ZERO_H = VECTOR2_ZERO;
__device__ __constant__ const float VECTOR2_ZERO_D[VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE] = VECTOR2_ZERO;
#define VECTOR2_ONE { 1, 1 }
const Vector2 VECTOR2_ONE_H = VECTOR2_ONE;
__device__ __constant__ const float VECTOR2_ONE_D[VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE] = VECTOR2_ONE;
#define VECTOR2_UP { 0, 1 }
const Vector2 VECTOR2_UP_H = VECTOR2_UP;
__device__ __constant__ const float VECTOR2_UP_D[VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE] = VECTOR2_UP;
#define VECTOR2_DOWN { 0, -1 }
const Vector2 VECTOR2_DOWN_H = VECTOR2_DOWN;
__device__ __constant__ const float VECTOR2_DOWN_D[VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE] = VECTOR2_DOWN;
#define VECTOR2_RIGHT { 1, 0 }
const Vector2 VECTOR2_RIGHT_H = VECTOR2_RIGHT;
__device__ __constant__ const float VECTOR2_RIGHT_D[VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE] = VECTOR2_RIGHT;
#define VECTOR2_LEFT { -1, 0 }
const Vector2 VECTOR2_LEFT_H = VECTOR2_LEFT;
__device__ __constant__ const float VECTOR2_LEFT_D[VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE] = VECTOR2_LEFT;

/** ------------------------------------ **/
#define VECTOR2_X_AXIS { 1, 0 }
const Vector2 VECTOR2_X_AXIS_H = VECTOR2_X_AXIS;
__device__ __constant__ const float VECTOR2_X_AXIS_D[VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE] = VECTOR2_X_AXIS;
#define VECTOR2_Y_AXIS { 0, 1 }
const Vector2 VECTOR2_Y_AXIS_H = VECTOR2_Y_AXIS;
__device__ __constant__ const float VECTOR2_Y_AXIS_D[VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE] = VECTOR2_Y_AXIS;
#undef VECTOR2_EQIVALENT_FLOAT_ARRAY_SIZE

/** ========================================================== **/
/*| GPU initialisation |*/
// Initializes constants on GPU
inline bool Vector2::initConstants() {
	if (cudaMemcpyToSymbol(VECTOR2_ZERO_D, &VECTOR2_ZERO_H, sizeof(Vector2), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(VECTOR2_ONE_D, &VECTOR2_ONE_H, sizeof(Vector2), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(VECTOR2_UP_D, &VECTOR2_UP_H, sizeof(Vector2), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(VECTOR2_DOWN_D, &VECTOR2_DOWN_H, sizeof(Vector2), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(VECTOR2_RIGHT_D, &VECTOR2_RIGHT_H, sizeof(Vector2), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(VECTOR2_LEFT_D, &VECTOR2_LEFT_H, sizeof(Vector2), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);

	if (cudaMemcpyToSymbol(VECTOR2_X_AXIS_D, &VECTOR2_X_AXIS_H, sizeof(Vector2), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);
	if (cudaMemcpyToSymbol(VECTOR2_Y_AXIS_D, &VECTOR2_Y_AXIS_H, sizeof(Vector2), 0, cudaMemcpyHostToDevice) != cudaSuccess) return(false);

	return(true);
}

/** ========================================================== **/
/*| Access |*/

/** ------------------------------------ **/
// Vector2(0, 0); Has visible constants VECTOR2_ZERO_H and *VECTOR2_ZERO_D for the host and the device, respectively, as well as macro VECTOR2_ZERO.
__dumb__ const Vector2& Vector2::zero() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_ZERO_D));
#else
	return VECTOR2_ZERO_H;
#endif
}
// Vector2(1, 1); Has visible constants VECTOR2_ONE_H and *VECTOR2_ONE_D for the host and the device, respectively, as well as macro VECTOR2_ONE.
__dumb__ const Vector2& Vector2::one() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_ONE_D));
#else
	return VECTOR2_ONE_H;
#endif
}
// Vector2(0, 1); Has visible constants VECTOR2_UP_H and *VECTOR2_UP_D for the host and the device, respectively, as well as macro VECTOR2_UP.
__dumb__ const Vector2& Vector2::up() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_UP_D));
#else
	return VECTOR2_UP_H;
#endif
}
// Vector2(0, -1); Has visible constants VECTOR2_DOWN_H and *VECTOR2_DOWN_D for the host and the device, respectively, as well as macro VECTOR2_DOWN.
__dumb__ const Vector2& Vector2::down() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_DOWN_D));
#else
	return VECTOR2_DOWN_H;
#endif
}
// Vector2(1, 0); Has visible constants VECTOR2_RIGHT_H and *VECTOR2_RIGHT_D for the host and the device, respectively, as well as macro VECTOR2_RIGHT.
__dumb__ const Vector2& Vector2::right() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_RIGHT_D));
#else
	return VECTOR2_RIGHT_H;
#endif
}
// Vector2(-1, 0); Has visible constants VECTOR2_LEFT_H and *VECTOR2_LEFT_D for the host and the device, respectively, as well as macro VECTOR2_LEFT.
__dumb__ const Vector2& Vector2::left() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_LEFT_D));
#else
	return VECTOR2_LEFT_H;
#endif
}

/** ------------------------------------ **/
// Vector2(1, 0); Has visible constants VECTOR2_X_AXIS_H and *VECTOR2_X_AXIS_D for the host and the device, respectively, as well as macro VECTOR2_X_AXIS.
__dumb__ const Vector2& Vector2::Xaxis() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_X_AXIS_D));
#else
	return VECTOR2_X_AXIS_H;
#endif
}
// Vector2(0, 1); Has visible constants VECTOR2_Y_AXIS_H and *VECTOR2_Y_AXIS_D for the host and the device, respectively, as well as macro VECTOR2_Y_AXIS.
__dumb__ const Vector2& Vector2::Yaxis() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_Y_AXIS_D));
#else
	return VECTOR2_Y_AXIS_H;
#endif
}
// Vector2(1, 0); Has visible constants VECTOR2_X_AXIS_H and *VECTOR2_X_AXIS_D for the host and the device, respectively, as well as macro VECTOR2_X_AXIS.
__dumb__ const Vector2& Vector2::i() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_X_AXIS_D));
#else
	return VECTOR2_X_AXIS_H;
#endif
}
// Vector2(0, 1); Has visible constants VECTOR2_Y_AXIS_H and *VECTOR2_Y_AXIS_D for the host and the device, respectively, as well as macro VECTOR2_Y_AXIS.
__dumb__ const Vector2& Vector2::j() {
#ifdef __CUDA_ARCH__
	return (*((Vector2*)VECTOR2_Y_AXIS_D));
#else
	return VECTOR2_Y_AXIS_H;
#endif
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Stream operators: **/
// Stream operator for input.
inline std::istream& operator>>(std::istream &stream, Vector2 &v) {
	return(stream >> v.x >> v.y);
}
// Stream operator for output.
inline std::ostream& operator<<(std::ostream &stream, const Vector2 &v) {
	return(stream << "(" << v.x << ", " << v.y << ")");
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Macro undefs: **/

/** ------------------------------------ **/
#undef SqrMagnitudeOfThisVector2
#undef MagnitudeOfThisVector2
#undef SqrMagnitudeVector2
#undef MagnitudeVector2

/** ------------------------------------ **/
#undef DotProductWithVector2
#undef DotProductVector2

/** ------------------------------------ **/
#undef CrossWith_Z_Vector2
#undef CrossProduct_Z_Vector2

/** ------------------------------------ **/
#undef AngleCosWithVector2
#undef AngleCosVector2

/** ------------------------------------ **/
#undef NormalOnVector2

/** ------------------------------------ **/
#undef DefCrossVector2
#undef AssignCrossWithVector2
#undef AssignCrossOfVector2
#undef DefCrossWithVector2
#undef DefCrossBetweenVector2

/** ------------------------------------ **/
#undef AssignAndReturnThisVector2
