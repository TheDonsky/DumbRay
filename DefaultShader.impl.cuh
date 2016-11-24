#include"DefaultShader.cuh"



__dumb__ DefaultShader::DefaultShader(ColorRGB color, float diffuse, float smoothness, float shine) {
	albedo = color;
	register float totalReflect = diffuse + smoothness;
	register float mul = ((totalReflect > 1.0f) ? (1.0f / totalReflect) : 1.0f);
	diff = diffuse * mul;
	gloss = smoothness * mul;
	shininess = shine;
}

__dumb__ ShaderReport DefaultShader::cast(const ShaderHitInfo<BakedTriFace> &input) {
	ShaderReport report;
	register Vector3 massCenter = input.object.vert.getMases(input.hitPoint);
	register Vector3 normal = input.object.norm.massCenter(massCenter);

	register Vector3 camDirection = (input.observer - input.hitPoint);
	register Vector3 reflectDirection = input.photon.ray.direction.reflection(normal);
	
	register ColorRGB color = input.photon.color * albedo;
	register float cosL = max(reflectDirection.angleCos(normal), 0.0f);
	register ColorRGB camColor = (color * (diff * cosL + gloss * pow(max(reflectDirection.angleCos(camDirection), 0.0f), shininess)));
	register ColorRGB reflColor = (color * (diff * cosL + gloss));

	report.observed = Photon(Ray(input.hitPoint, camDirection), camColor);
	report.reflection = Photon(Ray(input.hitPoint, reflectDirection), reflColor);
	report.refraction = Photon(Ray(Vector3::zero(), Vector3::zero()), ColorRGB(0, 0, 0));
	return report;
}
