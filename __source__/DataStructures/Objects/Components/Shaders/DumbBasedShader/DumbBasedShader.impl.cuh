#include "DumbBasedShader.cuh"



__dumb__ DumbBasedShader::DumbBasedShader(const ColorRGB &fresnelFactor, float cpecular, const ColorRGB &diffuse, float metal) {
	fres = fresnelFactor;
	metal = max(min(metal, 1.0f), 0.0f);
	diff = diffuse * (1.0f - metal);
	spec = cpecular;
	specMass = metal;
}


__dumb__ void DumbBasedShader::requestIndirectSamples(const ShaderInirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const {
	Vector3 sampleDir; 
	Vector3 normal = request.object->vert.normal().normalized();
	Vector3 n = request.object->norm.massCenter(request.object->vert.getMasses(request.hitPoint)).normalized();
	if ((n * (request.ray.direction)) >= 0.0f) return;
	Vector3 r = (request.ray.direction).reflection(n).normalized();
	if (r * normal < 0) normal *= -1;
	
	//Vector3 diffuseSample; request.context->entropy->pointOnSphere(diffuseSample.x, diffuseSample.y, diffuseSample.z);
	//if ((diffuseSample * normal) < 0.0f) diffuseSample *= -1;

	Vector3 a = r & Vector3(1.0f, 0.0f, 0.0f);
	if (normal.sqrMagnitude() < VECTOR_EPSILON) a = r & Vector3(0.0f, 1.0f, 0.0f);
	a.normalize();
	Vector3 b = (r & a);
	while (true) {
		float cosine = pow(request.context->entropy->getFloat(), 1.0f / spec);
		float sinus = sqrt(1.0f - (cosine * cosine));
		float angle = (request.context->entropy->getFloat() * 2.0f * PI);
		float angleCos = cos(angle);
		float angleSin = sqrt(1.0f - (angleCos * angleCos));
		sampleDir = ((r * cosine) + (sinus * ((angleCos * a) + (angleSin * b))));
		if ((sampleDir * normal) >= 0.0f) break;
	}

	//samples->set(SampleRay(Ray(request.hitPoint, ((specMass * sampleDir) + ((1.0f - specMass) * diffuseSample))), 1.0f));
	samples->set(SampleRay(Ray(request.hitPoint, sampleDir), 1.0f));
}
__dumb__ Color DumbBasedShader::getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const {
	Vector3 n = request.object->norm.massCenter(request.object->vert.getMasses(request.hitPoint)).normalized();
	if ((n * request.observerDirection) < 0.0f) return Color(0.0f, 0.0f, 0.0f);

	Vector3 wi = -request.photon.ray.direction.normalized();
	Vector3 w0 = request.observerDirection.normalized();
	Vector3 wh = (w0 + wi).normalized();
	Color fwi = Color(fresnel(fres.r, wh, wi), fresnel(fres.g, wh, wi), fresnel(fres.b, wh, wi));

	float dwh = ((spec + 2) / (2.0f * PI));
	if (request.photonType == PHOTON_TYPE_DIRECT_ILLUMINATION)
		dwh *= pow(n * wh, spec);

	float gwiw0 = min(1.0f, min(2.0f * (n * wh) * (n * w0) / (w0 * wh), 2.0f * (n * wh) * (n * wi) / (w0 * wh)));

	Color brdfBare = ((fwi * dwh * gwiw0) / (4.0f * (n * w0) * (n * wi)));
	
	Color brdf(
		max(0.0f, min(brdfBare.r, 1.0f)),
		max(0.0f, min(brdfBare.g, 1.0f)),
		max(0.0f, min(brdfBare.b, 1.0f)));

	Color diffuseColor = (diff * (n * wi));
	Color color(
		max(0.0f, min((brdf.r * specMass) + diffuseColor.r, 1.0f)),
		max(0.0f, min((brdf.g * specMass) + diffuseColor.g, 1.0f)),
		max(0.0f, min((brdf.b * specMass) + diffuseColor.b, 1.0f)));

	return (color * request.photon.color);
}


__dumb__ float DumbBasedShader::fresnel(float r, const Vector3 &wh, const Vector3 &wi) {
	register float val = (1.0f - (wh * wi));
	register float sqrVal = (val * val);
	return (r + ((1.0f - r) * (sqrVal * sqrVal * val)));
}
