#include"DefaultShader.cuh"



__device__ __host__ inline DefaultShader::DefaultShader(ColorRGB color, float diffuse, float smoothness, float shine) {
	albedo = color;
	register float totalReflect = diffuse + smoothness;
	register float mul = ((totalReflect > 1.0f) ? (1.0f / totalReflect) : 1.0f);
	diff = diffuse * mul;
	gloss = smoothness * mul;
	shininess = shine;
}

inline bool DefaultShader::uploadAt(DefaultShader *dst) {
	return (cudaMemcpy(dst, this, sizeof(DefaultShader), cudaMemcpyHostToDevice) == cudaSuccess);
}
inline bool DefaultShader::disposeOnDevice(DefaultShader *ptr) {
	return true;
}

__dumb__ Material<BakedTriFace>::ShaderReport DefaultShader::cast(const Material<BakedTriFace>::HitInfo &input) {
	Material<BakedTriFace>::ShaderReport report;
	register Vector3 massCenter = input.object.vert.getMases(input.hitPoint);
	register Vector3 normal = input.object.norm.massCenter(massCenter);

	register Vector3 camDirection = (input.observer - input.hitPoint);
	register Vector3 reflectDirection = input.photon.ray.direction.reflection(normal);
	
	register ColorRGB color = input.photon.color * albedo;
	register float cosL = max(reflectDirection.angleCos(normal), 0.0f);
	register ColorRGB camColor = (color * (diff * cosL + gloss * pow(max(reflectDirection.angleCos(camDirection), 0.0f), shininess)));
	register ColorRGB reflColor = (color * (diff * cosL + gloss));

	report.cameraPhoton = Photon(Ray(input.hitPoint, camDirection), camColor);
	report.reflectPhoton = Photon(Ray(input.hitPoint, reflectDirection), reflColor);
	return report;
}
