#pragma once
#include"Vector3.h"



struct Vector2 {
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Parameters: **/
	float x, y;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Construction: **/
	// Does nothing
	__dumb__ Vector2();
	// Simple constructor
	__dumb__ Vector2(float X, float Y);
	// Copy constructor
	__dumb__ Vector2(const Vector2 &v);
	// Cast constructor
	__dumb__ Vector2(const Vector3 &v);
	// Cast to Vector3
	__dumb__ operator Vector3()const;
	// Reconstructor
	__dumb__ Vector2& operator()(float X, float Y);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Operators: **/


	/** ========================================================== **/
	/*| + |*/
	// Enables writing "+v"
	__dumb__ Vector2 operator+()const;
	// Sum of two vectors
	__dumb__ Vector2 operator+(const Vector2 &v)const;
	// Increment
	__dumb__ Vector2& operator+=(const Vector2 &v);


	/** ========================================================== **/
	/*| - |*/
	// Inversed vector
	__dumb__ Vector2 operator-()const;
	// Subtraction of two vectors
	__dumb__ Vector2 operator-(const Vector2 &v)const;
	// Decrement
	__dumb__ Vector2& operator-=(const Vector2 &v);


	/** ========================================================== **/
	/*| * & ^ |*/

	/** ------------------------------------ **/
	// Multiplied by floating point
	__dumb__ Vector2 operator*(const float f)const;
	// Inversed syntax for floating point multiplication
	__dumb__ friend Vector2 operator*(const float f, const Vector2 &v);
	// Multilication by a floating point
	__dumb__ Vector2& operator*=(const float f);

	/** ------------------------------------ **/
	// Dot product
	__dumb__ float operator*(const Vector2 &v)const;

	/** ------------------------------------ **/
	// Z coordinate of the cross product
	__dumb__ float operator&(const Vector2 &v)const;

	/** ------------------------------------ **/
	// Upscaled (x*v.x, y*v.y)
	__dumb__ Vector2 operator^(const Vector2 &v)const;
	// Upscale
	__dumb__ Vector2& operator^=(const Vector2 &v);


	/** ========================================================== **/
	/*| / |*/

	/** ------------------------------------ **/
	// Divided by floating point
	__dumb__ Vector2 operator/(const float f)const;
	// Inversed syntax for floating point division(dumb and unnessessary, but anyway...)
	__dumb__ friend Vector2 operator/(const float f, const Vector2 &v);
	// Division by floating point
	__dumb__ Vector2& operator/=(const float f);

	/** ------------------------------------ **/
	// Downscaled (x/v.x, y/v.y)
	__dumb__ Vector2 operator/(const Vector2 &v)const;
	// Downscale
	__dumb__ Vector2& operator/=(const Vector2 &v);


	/** ========================================================== **/
	/*| == |*/
	// Compare (equals)
	__dumb__ bool operator==(const Vector2 &v)const;
	// Compare (not equals)
	__dumb__ bool operator!=(const Vector2 &v)const;
	// Compare (distance between is low enough)
	__dumb__ bool isNearTo(const Vector2 &v, const float maxDistance)const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Functions: **/


	/** ========================================================== **/
	/*| Magnitude |*/

	/** ------------------------------------ **/
	// Square of the vector's length
	__dumb__ float sqrMagnitude()const;
	// Vector's length
	__dumb__ float magnitude()const;

	/** ------------------------------------ **/
	// Distance to the other vertex
	__dumb__ float distanceTo(const Vector2 &v)const;
	// Distance between vertices
	__dumb__ static float distance(const Vector2 &v1, const Vector2 &v2);

	/** ------------------------------------ **/
	// Unit vector with the same direction
	__dumb__ Vector2 normalized()const;
	// Makes this vector unit, without changing direction
	__dumb__ Vector2& normalize();


	/** ========================================================== **/
	/*| Angle |*/

	/** ------------------------------------ **/
	// Cosine of angle between vectors
	__dumb__ static float angleCos(const Vector2 &v1, const Vector2 &v2);
	// Cosine of angle between unit vectors
	__dumb__ static float angleCosUnitVectors(const Vector2 &v1, const Vector2 &v2);
	// Cosine of angle between this and the another vector
	__dumb__ float angleCos(const Vector2 &v)const;
	// Cosine of angle between this and the another unit vector
	__dumb__ float angleCosUnitVectors(const Vector2 &v)const;

	/** ------------------------------------ **/
	// Sine of angle between vectors
	__dumb__ static float angleSin(const Vector2 &v1, const Vector2 &v2);
	// Sine of angle between unit vectors
	__dumb__ static float angleSinUnitVectors(const Vector2 &v1, const Vector2 &v2);
	// Sine of angle between this and the another vector
	__dumb__ float angleSin(const Vector2 &v)const;
	// Sine of angle between this and the another unit vector
	__dumb__ float angleSinUnitVectors(const Vector2 &v)const;

	/** ------------------------------------ **/
	// Angle between two vectors in radians
	__dumb__ static float radianAngle(const Vector2 &v1, const Vector2 &v2);
	// Angle between two unit vectors in radians
	__dumb__ static float radianAngleUnitVectors(const Vector2 &v1, const Vector2 &v2);
	// Angle between this and the another vector in radians
	__dumb__ float radianAngle(const Vector2 &v)const;
	// Angle between this and the another unit vector in radians
	__dumb__ float radianAngleUnitVectors(const Vector2 &v)const;

	/** ------------------------------------ **/
	// Angle between two vectors in degrees
	__dumb__ static float angle(const Vector2 &v1, const Vector2 &v2);
	// Angle between two unit vectors in degrees
	__dumb__ static float angleUnitVectors(const Vector2 &v1, const Vector2 &v2);
	// Angle between this and the another vector in degrees
	__dumb__ float angle(const Vector2 &v)const;
	// Angle between this and the another unit vector in degrees
	__dumb__ float angleUnitVectors(const Vector2 &v)const;


	/** ========================================================== **/
	/*| Normal |*/

	/** ------------------------------------ **/
	// Normal(projection) on another vector
	__dumb__ Vector2 normalOn(const Vector2 &v)const;
	// Normal(projection) on an unit vector
	__dumb__ Vector2 normalOnUnitVector(const Vector2 &v)const;
	// Projects on another vector
	__dumb__ Vector2& normalizeOn(const Vector2 &v);
	// Projects on an unit vector
	__dumb__ Vector2& normalizeOnUnitVector(const Vector2 &v);

	/** ------------------------------------ **/
	// Magnitude of the normal on the another vector
	__dumb__ float normalLengthOn(const Vector2 &v)const;
	// Magnitude of the normal on the unit vector
	__dumb__ float normalLengthOnUnitVector(const Vector2 &v)const;


	/** ========================================================== **/
	/*| Rotation |*/

	/** ------------------------------------ **/
	// This vector, rotated by given radian angle
	__dumb__ Vector2 rotated_Radian(float angle)const;
	// Rotates vector by given radian angle
	__dumb__ Vector2& rotate_Radian(float angle);

	/** ------------------------------------ **/
	// This vector, rotated by given angle in degrees
	__dumb__ Vector2 rotated(float angle)const;
	// Rotates vector by given angle in degrees
	__dumb__ Vector2& rotate(float angle);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Constants: **/

	/** ========================================================== **/
	/*| GPU initialisation |*/
	// Initializes constants on GPU
	inline static bool initConstants();

	/** ========================================================== **/
	/*| Access |*/

	/** ------------------------------------ **/
	// Vector2(0, 0); Has visible constants VECTOR2_ZERO_H and *VECTOR2_ZERO_D for the host and the device, respectively, as well as macro VECTOR2_ZERO.
	__dumb__ static const Vector2& zero();
	// Vector2(1, 1); Has visible constants VECTOR2_ONE_H and *VECTOR2_ONE_D for the host and the device, respectively, as well as macro VECTOR2_ONE.
	__dumb__ static const Vector2& one();
	// Vector2(0, 1); Has visible constants VECTOR2_UP_H and *VECTOR2_UP_D for the host and the device, respectively, as well as macro VECTOR2_UP.
	__dumb__ static const Vector2& up();
	// Vector2(0, -1); Has visible constants VECTOR2_DOWN_H and *VECTOR2_DOWN_D for the host and the device, respectively, as well as macro VECTOR2_DOWN.
	__dumb__ static const Vector2& down();
	// Vector2(1, 0); Has visible constants VECTOR2_RIGHT_H and *VECTOR2_RIGHT_D for the host and the device, respectively, as well as macro VECTOR2_RIGHT.
	__dumb__ static const Vector2& right();
	// Vector2(-1, 0); Has visible constants VECTOR2_LEFT_H and *VECTOR2_LEFT_D for the host and the device, respectively, as well as macro VECTOR2_LEFT.
	__dumb__ static const Vector2& left();

	/** ------------------------------------ **/
	// Vector2(1, 0); Has visible constants VECTOR2_X_AXIS_H and *VECTOR2_X_AXIS_D for the host and the device, respectively, as well as macro VECTOR2_X_AXIS.
	__dumb__ static const Vector2& Xaxis();
	// Vector2(0, 1); Has visible constants VECTOR2_Y_AXIS_H and *VECTOR2_Y_AXIS_D for the host and the device, respectively, as well as macro VECTOR2_Y_AXIS.
	__dumb__ static const Vector2& Yaxis();
	// Vector2(1, 0); Has visible constants VECTOR2_X_AXIS_H and *VECTOR2_X_AXIS_D for the host and the device, respectively, as well as macro VECTOR2_X_AXIS.
	__dumb__ static const Vector2& i();
	// Vector2(0, 1); Has visible constants VECTOR2_Y_AXIS_H and *VECTOR2_Y_AXIS_D for the host and the device, respectively, as well as macro VECTOR2_Y_AXIS.
	__dumb__ static const Vector2& j();





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Stream operators: **/
	// Stream operator for input.
	inline friend std::istream& operator>>(std::istream &stream, Vector2 &v);
	// Stream operator for output.
	inline friend std::ostream& operator<<(std::ostream &stream, const Vector2 &v);
};





#include"Vector2.impl.h"
