#pragma once
#include<algorithm>
#include"cuda_runtime.h"
#include<iostream>
#include<math.h>
#include"../../../../Namespaces/Dson/Dson.h"

#define PI 3.14159265359f
#define RADIAN 57.29577951308233f
#define VECTOR_EPSILON 0.000005f
#define EPSILON_VECTOR Vector3(VECTOR_EPSILON, VECTOR_EPSILON, VECTOR_EPSILON)

#ifndef __device__
#define __dumb__ inline 
#else
#define __dumb__ __device__ __host__ inline
#endif

/*
	This structure represents arbitrary three dimensional vector or a vertex.
	The structure provides with most of the functionality, one might be epecting
	from a three dimensional vector.
	PS: this can be used interchangeably with typename "Vertex".
 */
struct Vector3 : float3 {
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Construction: **/
	// Does nothing
	__dumb__ Vector3();
	// Simple constructor
	__dumb__ Vector3(float X, float Y, float Z);
	// Copy constructor
	__dumb__ Vector3(const Vector3 &v);
	// Reconstructor
	__dumb__ Vector3& operator()(float X, float Y, float Z);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Operators: **/

	
	/** ========================================================== **/
	/*| + |*/
	// Enables writing "+v"
	__dumb__ Vector3 operator+()const;
	// Sum of two vectors
	__dumb__ Vector3 operator+(const Vector3 &v)const;
	// Increment
	__dumb__ Vector3& operator+=(const Vector3 &v);

	
	/** ========================================================== **/
	/*| - |*/
	// Inversed vector
	__dumb__ Vector3 operator-()const;
	// Subtraction of two vectors
	__dumb__ Vector3 operator-(const Vector3 &v)const;
	// Decrement
	__dumb__ Vector3& operator-=(const Vector3 &v);

	
	/** ========================================================== **/
	/*| * & ^ |*/
	
	/** ------------------------------------ **/
	// Multiplied by floating point
	__dumb__ Vector3 operator*(const float f)const;
	// Inversed syntax for floating point multiplication
	__dumb__ friend Vector3 operator*(const float f, const Vector3 &v);
	// Multilication by a floating point
	__dumb__ Vector3& operator*=(const float f);
	
	/** ------------------------------------ **/
	// Dot product
	__dumb__ float operator*(const Vector3 &v)const;
	
	/** ------------------------------------ **/
	// Cross product
	__dumb__ Vector3 operator&(const Vector3 &v)const;
	// Sets vector to it's cross product with the another one
	__dumb__ Vector3& operator&=(const Vector3 &v);
	
	/** ------------------------------------ **/
	// Upscaled (x*v.x, y*v.y, z*v.z)
	__dumb__ Vector3 operator^(const Vector3 &v)const;
	// Upscale
	__dumb__ Vector3& operator^=(const Vector3 &v);


	/** ========================================================== **/
	/*| / |*/
	
	/** ------------------------------------ **/
	// Divided by floating point
	__dumb__ Vector3 operator/(const float f)const;
	// Inversed syntax for floating point division(dumb and unnessessary, but anyway...)
	__dumb__ friend Vector3 operator/(const float f, const Vector3 &v);
	// Division by floating point
	__dumb__ Vector3& operator/=(const float f);
	
	/** ------------------------------------ **/
	// Downscaled (x/v.x, y/v.y, z/v.y)
	__dumb__ Vector3 operator/(const Vector3 &v)const;
	// Downscale
	__dumb__ Vector3& operator/=(const Vector3 &v);


	/** ========================================================== **/
	/*| == |*/
	// Compare (equals)
	__dumb__ bool operator==(const Vector3 &v)const;
	// Compare (not equals)
	__dumb__ bool operator!=(const Vector3 &v)const;
	// Compare (distance between is low enough)
	__dumb__ bool isNearTo(const Vector3 &v, const float maxDistance)const;





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
	__dumb__ float distanceTo(const Vector3 &v)const;
	// Distance between vertices
	__dumb__ static float distance(const Vector3 &v1, const Vector3 &v2);
	
	/** ------------------------------------ **/
	// Unit vector with the same direction
	__dumb__ Vector3 normalized()const;
	// Makes this vector unit, without changing direction
	__dumb__ Vector3& normalize();


	/** ========================================================== **/
	/*| Angle |*/
	
	/** ------------------------------------ **/
	// Cosine of angle between vectors
	__dumb__ static float angleCos(const Vector3 &v1, const Vector3 &v2);
	// Cosine of angle between unit vectors
	__dumb__ static float angleCosUnitVectors(const Vector3 &v1, const Vector3 &v2);
	// Cosine of angle between this and the another vector
	__dumb__ float angleCos(const Vector3 &v)const;
	// Cosine of angle between this and the another unit vector
	__dumb__ float angleCosUnitVectors(const Vector3 &v)const;

	/** ------------------------------------ **/
	// Sine of angle between vectors
	__dumb__ static float angleSin(const Vector3 &v1, const Vector3 &v2);
	// Sine of angle between unit vectors
	__dumb__ static float angleSinUnitVectors(const Vector3 &v1, const Vector3 &v2);
	// Sine of angle between this and the another vector
	__dumb__ float angleSin(const Vector3 &v)const;
	// Sine of angle between this and the another unit vector
	__dumb__ float angleSinUnitVectors(const Vector3 &v)const;

	/** ------------------------------------ **/
	// Angle between two vectors in radians
	__dumb__ static float radianAngle(const Vector3 &v1, const Vector3 &v2);
	// Angle between two unit vectors in radians
	__dumb__ static float radianAngleUnitVectors(const Vector3 &v1, const Vector3 &v2);
	// Angle between this and the another vector in radians
	__dumb__ float radianAngle(const Vector3 &v)const;
	// Angle between this and the another unit vector in radians
	__dumb__ float radianAngleUnitVectors(const Vector3 &v)const;

	/** ------------------------------------ **/
	// Angle between two vectors in degrees
	__dumb__ static float angle(const Vector3 &v1, const Vector3 &v2);
	// Angle between two unit vectors in degrees
	__dumb__ static float angleUnitVectors(const Vector3 &v1, const Vector3 &v2);
	// Angle between this and the another vector in degrees
	__dumb__ float angle(const Vector3 &v)const;
	// Angle between this and the another unit vector in degrees
	__dumb__ float angleUnitVectors(const Vector3 &v)const;


	/** ========================================================== **/
	/*| Normal |*/
	
	/** ------------------------------------ **/
	// Normal(projection) on another vector
	__dumb__ Vector3 normalOn(const Vector3 &v)const;
	// Normal(projection) on an unit vector
	__dumb__ Vector3 normalOnUnitVector(const Vector3 &v)const;
	// Projects on another vector
	__dumb__ Vector3& normalizeOn(const Vector3 &v);
	// Projects on an unit vector
	__dumb__ Vector3& normalizeOnUnitVector(const Vector3 &v);

	/** ------------------------------------ **/
	// Magnitude of the normal on the another vector
	__dumb__ float normalLengthOn(const Vector3 &v)const;
	// Magnitude of the normal on the unit vector
	__dumb__ float normalLengthOnUnitVector(const Vector3 &v)const;

	/** ------------------------------------ **/
	// Noraml of the plain, defined by two vectors
	__dumb__ static Vector3 plainNormal(const Vector3 &v1, const Vector3 &v2);
	// Noraml of the plain, defined by two unit vectors
	__dumb__ static Vector3 plainNormalForUnitVectors(const Vector3 &v1, const Vector3 &v2);
	// Noraml of the plain, defined by three vertices in space
	__dumb__ static Vector3 plainNormal(const Vector3 &v1, const Vector3 &v2, const Vector3 &v3);

	/** ------------------------------------ **/
	// Reflection
	__dumb__ Vector3 reflection(const Vector3 &normal)const;
	__dumb__ Vector3& reflect(const Vector3 &normal);
	__dumb__ Vector3 reflectionOnUnitVector(const Vector3 &normal)const;
	__dumb__ Vector3& reflectOnUnitVector(const Vector3 &normal);


	/** ========================================================== **/
	/*| Rotation |*/
	
	/** ------------------------------------ **/
	// This vector, rotated against given vector by given radian angle
	__dumb__ Vector3 rotated_Radian(const Vector3 &axis, float angle)const;
	// This vector, rotated against given axis by given radian angle, if both vectors are unit and perpendicular to each other
	__dumb__ Vector3 rotated_AxisAgainstAxisRadian(const Vector3 &axis, float angle)const;
	// Rotates vector against given vector by given radian angle
	__dumb__ Vector3& rotate_Radian(const Vector3 &axis, float angle);
	// Rotates vector against given vector by given radian angle, if both vectors are unit and perpendicular to each other
	__dumb__ Vector3& rotate_AxisAgainstAxisRadian(const Vector3 &axis, float angle);

	/** ------------------------------------ **/
	// This vector, rotated against given vector by given angle in degrees
	__dumb__ Vector3 rotated(const Vector3 &axis, float angle)const;
	// This vector, rotated against given axis by given angle in degrees, if both vectors are unit and perpendicular to each other
	__dumb__ Vector3 rotated_AxisAgainstAxis(const Vector3 &axis, float angle)const;
	// Rotates vector against given vector by given angle in degrees
	__dumb__ Vector3& rotate(const Vector3 &axis, float angle);
	// Rotates vector against given vector by given angle in degrees, if both vectors are unit and perpendicular to each other
	__dumb__ Vector3& rotate_AxisAgainstAxis(const Vector3 &axis, float angle);

	/** ------------------------------------ **/
	// Rotates two vectors by given radian angle against itself, if all three vectors are unit and perpendicular to each other
	__dumb__ void rotateTwoAxisAgainstThisRadian(Vector3 &v1, Vector3 &v2, float angle)const;
	// Rotates two vectors by given angle in degrees against itself, if all three vectors are unit and perpendicular to each other
	__dumb__ void rotateTwoAxisAgainstThis(Vector3 &v1, Vector3 &v2, float angle)const;





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
	// Vector3(0, 0, 0); Has visible constants VECTOR3_ZERO_H and *VECTOR3_ZERO_D for the host and the device, respectively, as well as macro VECTOR3_ZERO.
	__dumb__ static const Vector3& zero();
	// Vector3(1, 1, 1); Has visible constants VECTOR3_ONE_H and *VECTOR3_ONE_D for the host and the device, respectively, as well as macro VECTOR3_ONE.
	__dumb__ static const Vector3& one();
	// Vector3(0, 1, 0); Has visible constants VECTOR3_UP_H and *VECTOR3_UP_D for the host and the device, respectively, as well as macro VECTOR3_UP.
	__dumb__ static const Vector3& up();
	// Vector3(0, -1, 0); Has visible constants VECTOR3_DOWN_H and *VECTOR3_DOWN_D for the host and the device, respectively, as well as macro VECTOR3_DOWN.
	__dumb__ static const Vector3& down();
	// Vector3(0, 0, 1); Has visible constants VECTOR3_FRONT_H and *VECTOR3_FRONT_D for the host and the device, respectively, as well as macro VECTOR3_FRONT.
	__dumb__ static const Vector3& front();
	// Vector3(0, 0, -1); Has visible constants VECTOR3_BACK_H and *VECTOR3_BACK_D for the host and the device, respectively, as well as macro VECTOR3_BACK.
	__dumb__ static const Vector3& back();
	// Vector3(1, 0, 0); Has visible constants VECTOR3_RIGHT_H and *VECTOR3_RIGHT_D for the host and the device, respectively, as well as macro VECTOR3_RIGHT.
	__dumb__ static const Vector3& right();
	// Vector3(-1, 0, 0); Has visible constants VECTOR3_LEFT_H and *VECTOR3_LEFT_D for the host and the device, respectively, as well as macro VECTOR3_LEFT.
	__dumb__ static const Vector3& left();

	/** ------------------------------------ **/
	// Vector3(1, 0, 0); Has visible constants VECTOR3_X_AXIS_H and *VECTOR3_X_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_X_AXIS.
	__dumb__ static const Vector3& Xaxis();
	// Vector3(0, 1, 0); Has visible constants VECTOR3_Y_AXIS_H and *VECTOR3_Y_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_Y_AXIS.
	__dumb__ static const Vector3& Yaxis();
	// Vector3(0, 0, 1); Has visible constants VECTOR3_Z_AXIS_H and *VECTOR3_Z_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_Z_AXIS.
	__dumb__ static const Vector3& Zaxis();
	// Vector3(1, 0, 0); Has visible constants VECTOR3_X_AXIS_H and *VECTOR3_X_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_X_AXIS.
	__dumb__ static const Vector3& i();
	// Vector3(0, 1, 0); Has visible constants VECTOR3_Y_AXIS_H and *VECTOR3_Y_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_Y_AXIS.
	__dumb__ static const Vector3& j();
	// Vector3(0, 0, 1); Has visible constants VECTOR3_Z_AXIS_H and *VECTOR3_Z_AXIS_D for the host and the device, respectively, as well as macro VECTOR3_Z_AXIS.
	__dumb__ static const Vector3& k();





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Stream operators: **/
	// Stream operator for input.
	inline friend std::istream& operator>>(std::istream &stream, Vector3 &v);
	// Stream operator for output.
	inline friend std::ostream& operator<<(std::ostream &stream, const Vector3 &v);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** From/to Dson: **/
	inline bool fromDson(const Dson::Object &object, std::ostream *errorStream);
	inline Dson::Array toDson()const;
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Alternative name for Vector3: **/
// This is adviced to be used, when we treat Vector3 like a point in space and not an actual vector. Other than that, the two are one and the same.
typedef Vector3 Vertex;





#include"Vector3.impl.h"
