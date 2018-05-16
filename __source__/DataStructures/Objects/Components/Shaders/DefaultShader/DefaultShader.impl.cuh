#include"DefaultShader.cuh"


template<typename HitType>
__dumb__ DefaultShaderGeneric<HitType>::DefaultShaderGeneric(ColorRGB color, float diffuse, float smoothness, float shine) {
	albedo = color;
	register float totalReflect = diffuse + smoothness;
	register float mul = ((totalReflect > 1.0f) ? (1.0f / totalReflect) : 1.0f);
	diff = diffuse * mul;
	gloss = smoothness * mul;
	shininess = shine;
}

template<typename HitType>
__dumb__ DefaultShaderGeneric<HitType>::ShaderReport DefaultShaderGeneric<HitType>::cast(const ShaderHitInfo &input)const {
	ShaderReport report;
	const HitType &object = (*input.object);
	register Vector3 massCenter = object.vert.getMasses(input.hitPoint);
	register Vector3 normal = object.norm.massCenter(massCenter);

	register Vector3 camDirection = (input.observer - input.hitPoint);
	register Vector3 reflectDirection = input.photon.ray.direction.reflection(normal);
	
	register ColorRGB color = input.photon.color * albedo;
	register float cosL = max(reflectDirection.angleCos(normal), 0.0f);
	register ColorRGB camColor = (color * (diff * cosL + gloss * pow(max(reflectDirection.angleCos(camDirection), 0.0f), shininess)));
	register ColorRGB reflColor = (color * gloss);

	report.observed = Photon(Ray(input.hitPoint, camDirection), camColor);
	report.bounce = Photon(Ray(input.hitPoint, reflectDirection), reflColor);
	return report;
}

template<typename HitType>
__dumb__ void DefaultShaderGeneric<HitType>::requestIndirectSamples(const ShaderInirectSamplesRequest<HitType> &request, RaySamples *samples)const {
	ShaderHitInfo info = {
		request.object,
		Photon(request.ray, Color(1.0f, 1.0f, 1.0f, 1.0f)),
		request.hitPoint,
		request.hitPoint + Vector3(1.0f, 1.0f, 1.0f) };
	ShaderReport report = cast(info);
	samples->set(SampleRay(report.bounce.ray, 1.0f));
}
template<typename HitType>
__dumb__ Color DefaultShaderGeneric<HitType>::getReflectedColor(const ShaderReflectedColorRequest<HitType> &request)const {
	ShaderHitInfo info = { 
		request.object, 
		request.photon,
		request.hitPoint, 
		request.hitPoint + request.observerDirection };
	return cast(info).observed.color;
}

