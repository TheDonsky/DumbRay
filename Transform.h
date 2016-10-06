#pragma once

#include"Vector3.h"

class Transform{
public:
	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Construction: **/
	__device__ __host__ inline Transform(const Vector3 &pos = Vector3(0, 0, 0), const Vector3 &euler = Vector3(0, 0, 0), const Vector3 &scl = Vector3(1, 1, 1));
	




	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Getters: **/
	__device__ __host__ inline const Vector3& getPosition()const;
	__device__ __host__ inline const Vector3& getEulerAngles()const;
	__device__ __host__ inline const Vector3& getScale()const;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Setters: **/
	__device__ __host__ inline Transform& setPosition(const Vector3 &newPos);
	__device__ __host__ inline Transform& move(const Vector3 &posDelta);

	__device__ __host__ inline Transform& setEulerAngles(const Vector3 &newEuler);
	__device__ __host__ inline Transform& rotate(const Vector3 &eulerDelta);

	__device__ __host__ inline Transform& setScale(const Vector3 &newScl);
	__device__ __host__ inline Transform& upscale(const Vector3 &sclDelta);





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Transform & detransform: **/
	__device__ __host__ inline friend const Vector3& operator>>=(Vector3 &v, const Transform &t);
	__device__ __host__ inline friend Vector3 operator>>(const Vector3 &v, const Transform &t);
	__device__ __host__ inline friend const Vector3& operator<<=(Vector3 &v, const Transform &t);
	__device__ __host__ inline friend Vector3 operator<<(const Vector3 &v, const Transform &t);





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
