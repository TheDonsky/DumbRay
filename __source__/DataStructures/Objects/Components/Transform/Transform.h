#pragma once

#include"../../../Primitives/Pure/Vector3/Vector3.h"
#include"../../../Primitives/Compound/Ray/Ray.h"

class Transform{
public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Construction: **/
	// Constructor
	__device__ __host__ inline Transform(const Vector3 &pos = Vector3(0, 0, 0), const Vector3 &euler = Vector3(0, 0, 0), const Vector3 &scl = Vector3(1, 1, 1));
	




	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Getters: **/
	// Transform position
	__device__ __host__ inline const Vector3& getPosition()const;
	// Transform euler angles
	__device__ __host__ inline const Vector3& getEulerAngles()const;
	// Transform scale
	__device__ __host__ inline const Vector3& getScale()const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Setters: **/
	// Sets tranform position
	__device__ __host__ inline Transform& setPosition(const Vector3 &newPos);
	// Moves transform (translates position)
	__device__ __host__ inline Transform& move(const Vector3 &posDelta);

	// Sets transform euler angles
	__device__ __host__ inline Transform& setEulerAngles(const Vector3 &newEuler);
	// Rotates transform
	__device__ __host__ inline Transform& rotate(const Vector3 &eulerDelta);

	// Sets transform scale
	__device__ __host__ inline Transform& setScale(const Vector3 &newScl);
	// Scales transform
	__device__ __host__ inline Transform& upscale(const Vector3 &sclDelta);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Transform & detransform: **/

	/** ========================================================== **/
	/*| Vectors |*/
	// Transforms Vertex
	__device__ __host__ inline friend Vector3& operator>>=(Vector3 &v, const Transform &t);
	// Transformed Vertex
	__device__ __host__ inline friend Vector3 operator>>(const Vector3 &v, const Transform &t);
	// Detransforms Vertex
	__device__ __host__ inline friend Vector3& operator<<=(Vector3 &v, const Transform &t);
	// Detransformed Vertex
	__device__ __host__ inline friend Vector3 operator<<(const Vector3 &v, const Transform &t);
	
	/** ========================================================== **/
	/*| Rays |*/
	// Transforms Ray
	__device__ __host__ inline friend Ray operator>>(const Ray &r, const Transform &trans);
	// Transformed Ray
	__device__ __host__ inline friend Ray& operator>>=(Ray &r, const Transform &trans);
	// Detransforms Ray
	__device__ __host__ inline friend Ray operator<<(const Ray &r, const Transform &trans);
	// Detransformed Ray
	__device__ __host__ inline friend Ray& operator<<=(Ray &r, const Transform &trans);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Directions: **/

	/** ========================================================== **/
	/*| Vectors |*/
	// Forward
	__device__ __host__ inline Vector3 front()const;
	// Back
	__device__ __host__ inline Vector3 back()const;
	// Right
	__device__ __host__ inline Vector3 right()const;
	// Left
	__device__ __host__ inline Vector3 left()const;
	// Up
	__device__ __host__ inline Vector3 up()const;
	// Down
	__device__ __host__ inline Vector3 down()const;
	// Direction
	__device__ __host__ inline Vector3 direction(Vector3 dir)const;
	
	/** ========================================================== **/
	/*| Rays |*/
	// Forward
	__device__ __host__ inline Ray frontRay()const;
	// Back
	__device__ __host__ inline Ray backRay()const;
	// Right
	__device__ __host__ inline Ray rightRay()const;
	// Left
	__device__ __host__ inline Ray leftRay()const;
	// Up
	__device__ __host__ inline Ray upRay()const;
	// Down
	__device__ __host__ inline Ray downRay()const;
	// Direction
	__device__ __host__ inline Ray ray(Vector3 dir)const;





private:
	struct TransformCoordinateSystem{
		Vector3 x, y, z;
	};
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Parameters: **/
	Vector3 position, eulerAngles, scale;
	TransformCoordinateSystem trans, detrans;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** System rebake: **/
	__device__ __host__ inline void rebake();
};


#include"Transform.impl.h"
