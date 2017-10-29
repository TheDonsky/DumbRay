#pragma once
#include"../../../GeneralPurpose/Generic/Generic.cuh"
#include"../../../Primitives/Pure/Vector3/Vector3.h"


/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename ElemType>
/*
Output of Raycaster::cast
*/
struct RaycastHit {
	// Object, the ray hit
	const ElemType *object;
	// Distance, the ray traveled before hitting the object
	float hitDistance;
	// Collision point
	Vector3 hitPoint;

	// Default constructor (does nothing)
	__device__ __host__ inline RaycastHit();
	// Constructs RaycastHit from the given parameters
	__device__ __host__ inline RaycastHit(const ElemType &elem, const float d, const Vector3 &p);
	// Constructs RaycastHit from the given parameters
	__device__ __host__ inline RaycastHit& operator()(const ElemType &elem, const float d, const Vector3 &p);
	// Constructs RaycastHit from the given parameters
	__device__ __host__ inline void set(const ElemType &elem, const float d, const Vector3 &p);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
class RaycastFunctionPack {
public:
	__dumb__ RaycastFunctionPack();

	__dumb__ void clean();
	template<typename RaycasterType>
	__dumb__ void use();

	typedef bool(*CastBreaker)(RaycastHit<HitType> &hit, const Ray &ray, bool &rv);
	// Casts a ray (returns true if the ray hits something; result is written in hit)
	__dumb__ bool cast(const void *raycaster, const Ray &r, RaycastHit<HitType> &hit, bool clipBackfaces, CastBreaker castBreaker)const;


private:
	bool(*castFunction)(const void *raycaster, const Ray &r, RaycastHit<HitType> &hit, bool clipBackfaces, CastBreaker castBreaker);
	template<typename RaycasterType>
	__dumb__ static bool castGeneric(const void *raycaster, const Ray &r, RaycastHit<HitType> &hit, bool clipBackfaces, CastBreaker castBreaker);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType> class Raycaster;
template<typename HitType>
class TypeTools<Raycaster<HitType> > {
public:
	typedef Raycaster<HitType> RaycasterType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(RaycasterType);
};


template<typename HitType> 
class Raycaster : public Generic<RaycastFunctionPack<HitType> > {
public:
	typedef bool(*CastBreaker)(RaycastHit<HitType> &hit, const Ray &ray, bool &rv);
	// Casts a ray (returns true if the ray hits something; result is written in hit)
	__dumb__ bool cast(const Ray &r, RaycastHit<HitType> &hit, bool clipBackfaces = false, CastBreaker castBreaker = NULL)const;

	inline Raycaster *upload()const;
	inline static Raycaster* upload(const Raycaster *source, int count = 1);
};


#include"Raycaster.impl.cuh"
