#pragma once
#include"../../../Primitives/Compound/Photon/Photon.cuh"
#include"../../../GeneralPurpose/Generic/Generic.cuh"
#include"../../Components/DumbStructs.cuh"
#include"../../../../Namespaces/Shapes/Shapes.cuh"
#include"../../Scene/Raycasters/Raycaster.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
struct ShaderInirectSamplesRequest {
	// Object, that was hit
	const HitType *object;

	// Ray, that hit the object
	Ray ray;

	// Distance to the hit point
	float hitDistance; 
	
	// Hit point
	Vector3 hitPoint;
	
	// Relative contribution of all output samples
	float absoluteSampleWeight;
};

template<typename HitType>
struct ShaderReflectedColorRequest {
	// Object, that was hit
	const HitType *object;
	
	// Photon, that hit the object
	Photon photon;

	// Hit point
	Vector3 hitPoint;

	// The direction to the observer:
	// (The result will be interpreted as Photon(Ray(hitPoint, observerDirection), Color(whatever the shader returns)) towards the observer)
	Vector3 observerDirection;
};




/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
class Shader {
public:
	__dumb__ void clean();
	template<typename ShaderType>
	__dumb__ void use();

	__dumb__ void requestIndirectSamples(const void *shader, const ShaderInirectSamplesRequest<HitType> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const void *shader, const ShaderReflectedColorRequest<HitType> &request)const;


private:
	void(*requestIndirectSamplesFn)(const void *shader, const ShaderInirectSamplesRequest<HitType> &request, RaySamples *samples);
	Color(*getReflectedColorFn)(const void *shader, const ShaderReflectedColorRequest<HitType> &request);


	template<typename ShaderType>
	__dumb__ static void requestIndirectSamplesGeneric(const void *shader, const ShaderInirectSamplesRequest<HitType> &request, RaySamples *samples);
	template<typename ShaderType>
	__dumb__ static Color getReflectedColorGeneric(const void *shader, const ShaderReflectedColorRequest<HitType> &request);
};




/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType> class Material;
template<typename HitType>
class TypeTools<Material<HitType> > {
public:
	typedef Material<HitType> MaterialType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(MaterialType);
};



template<typename HitType>
class Material : public Generic<Shader<HitType> > {
public:
	__dumb__ void requestIndirectSamples(const ShaderInirectSamplesRequest<HitType> &request, RaySamples *samples)const;
	__dumb__ Color getReflectedColor(const ShaderReflectedColorRequest<HitType> &request)const;


	inline Material *upload()const;
	inline static Material* upload(const Material *source, int count = 1);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType> struct Renderable;
template<typename HitType>
class TypeTools<Renderable<HitType> > {
public:
	typedef Renderable<HitType> RenderableType;
	DEFINE_TYPE_TOOLS_CONTENT_FOR(RenderableType);
};
template<typename HitType>
struct Renderable {
	HitType object;
	int materialId;

	__dumb__ Renderable();
	__dumb__ Renderable(const HitType &obj, int matId);

	__dumb__ bool intersects(const Renderable &other)const;
	__dumb__ bool intersects(const AABB &other)const;
	__dumb__ bool cast(const Ray& r, bool clipBackface)const;
	__dumb__ bool castPreInversed(const Ray& inversedRay, bool clipBackface)const;
	__dumb__ bool cast(const Ray& ray, float &hitDistance, Vertex& hitPoint, bool clipBackface)const;
	template<typename BoundType>
	__dumb__ bool sharesPoint(const Renderable& b, const BoundType& commonPointBounds)const;
	template<typename Shape>
	__dumb__ Vertex intersectionCenter(const Shape &shape)const;
	template<typename Shape>
	__dumb__ AABB intersectionBounds(const Shape &shape)const;
	__dumb__ Vertex massCenter()const;
	__dumb__ AABB boundingBox()const;
	__dumb__ void dump()const;
};





#include"Material.impl.cuh"
