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
__dumb__ ShaderReport DefaultShaderGeneric<HitType>::cast(const ShaderHitInfo<HitType> &input)const {
	ShaderReport report;
	register Vector3 massCenter = input.object.vert.getMases(input.hitPoint);
	register Vector3 normal = input.object.norm.massCenter(massCenter);

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
__dumb__ void DefaultShaderGeneric<HitType>::bounce(const ShaderBounceInfo<HitType> &info, ShaderBounce *bounce)const {
	if (bounce == NULL) return;
	ShaderHitInfo<HitType> castInfo = { info.object, info.photon, info.hitPoint, info.hitPoint - info.photon.ray.direction };
	ShaderReport report = cast(castInfo);
	bounce->samples[0] = report.bounce;
	bounce->count = 1;
}
template<typename HitType>
__dumb__ Photon DefaultShaderGeneric<HitType>::illuminate(const ShaderHitInfo<HitType>& info)const {
	return cast(info).observed;
}
