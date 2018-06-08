#include "DumbBasedShader.cuh"



__dumb__ DumbBasedShader::DumbBasedShader(const ColorRGB &fresnelFactor, float cpecular, const ColorRGB &diffuse, float metal) {
	fres = fresnelFactor;
	metal = max(min(metal, 1.0f), 0.0f);
	diff = diffuse * (1.0f - metal);
	spec = cpecular;
	specMass = metal;
}


__dumb__ void DumbBasedShader::requestIndirectSamples(const ShaderIndirectSamplesRequest<BakedTriFace> &request, RaySamples *samples)const {
	Vector3 sampleDir; 
	uint32_t sampleType;
	Vector3 normal = request.object->vert.normal().normalized();
	if ((normal * (request.ray.direction)) > 0.0f) {
		samples->set(SampleRay(Ray(request.hitPoint - (request.ray.direction * (8.0f * VECTOR_EPSILON)), request.ray.direction), 1.0f, request.significance, 0));
	}

	if (request.context->entropy->getBool(specMass)) {
		Vector3 n = request.object->norm.massCenter(request.object->vert.getMasses(request.hitPoint)).normalized();
		if ((n * (request.ray.direction)) >= 0.0f) return;
		Vector3 r = (request.ray.direction).reflection(n).normalized();
		if (r * normal < 0) normal *= -1;

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
		sampleType = 1;
	}
	else {
		request.context->entropy->pointOnSphere(sampleDir.x, sampleDir.y, sampleDir.z);
		if ((sampleDir * normal) < 0.0f) sampleDir *= -1;
		sampleType = 2;
	}

	samples->set(SampleRay(Ray(request.hitPoint, sampleDir), 1.0f, request.significance, sampleType));
}
__dumb__ Color DumbBasedShader::getReflectedColor(const ShaderReflectedColorRequest<BakedTriFace> &request)const {
	if ((request.object->vert.normal().normalized() * (request.photon.ray.direction)) > 0.0f)
		if (request.photonType != PHOTON_TYPE_DIRECT_ILLUMINATION)
			return request.photon.color;
	Vector3 n = request.object->norm.massCenter(request.object->vert.getMasses(request.hitPoint)).normalized();
	if ((n * request.observerDirection) < 0.0f) return Color(0.0f, 0.0f, 0.0f);
	Vector3 wi = -request.photon.ray.direction.normalized();
	
	Color brdf;
	bool brdfMatters = (request.photonType == PHOTON_TYPE_DIRECT_ILLUMINATION || request.sampleType == 1);
	if (brdfMatters) {
		Vector3 w0 = request.observerDirection.normalized();
		Vector3 wh = (w0 + wi).normalized();
		Color fwi = Color(fresnel(fres.r, wh, wi), fresnel(fres.g, wh, wi), fresnel(fres.b, wh, wi));

		float dwh = ((spec + 2) / (2.0f * PI));
		if (request.photonType == PHOTON_TYPE_DIRECT_ILLUMINATION)
			dwh *= pow(n * wh, spec);

		float gwiw0 = min(1.0f, min(2.0f * (n * wh) * (n * w0) / (w0 * wh), 2.0f * (n * wh) * (n * wi) / (w0 * wh)));

		Color brdfBare = ((fwi * dwh * gwiw0) / (4.0f * (n * w0) * (n * wi)));

		brdf = Color(
			max(0.0f, min(brdfBare.r, 1.0f)),
			max(0.0f, min(brdfBare.g, 1.0f)),
			max(0.0f, min(brdfBare.b, 1.0f)));
	}
	else brdf = Color(0.0f, 0.0f, 0.0f);

	Color diffuseColor;
	bool diffuseColorMatters = (request.photonType == PHOTON_TYPE_DIRECT_ILLUMINATION || request.sampleType == 2);
	if (diffuseColorMatters) diffuseColor = (diff * (n * wi));
	else diffuseColor = Color(0.0f, 0.0f, 0.0f);
	
	Color color;
	if (brdfMatters && diffuseColorMatters) color = Color(
		max(0.0f, min((brdf.r * specMass) + diffuseColor.r, 1.0f)),
		max(0.0f, min((brdf.g * specMass) + diffuseColor.g, 1.0f)),
		max(0.0f, min((brdf.b * specMass) + diffuseColor.b, 1.0f)));
	else if (brdfMatters) color = brdf;
	else color = (diffuseColor * (1.0f / (1.0f - specMass)));
	
	return (color * request.photon.color);
}


__dumb__ float DumbBasedShader::fresnel(float r, const Vector3 &wh, const Vector3 &wi) {
	register float val = (1.0f - (wh * wi));
	register float sqrVal = (val * val);
	return (r + ((1.0f - r) * (sqrVal * sqrVal * val)));
}


inline bool DumbBasedShader::fromDson(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type() != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Error: DumbBasedShader can only accept Dson::Dict in fromDson method..." << std::endl;
		return false;
	}
	const Dson::Dict &dict = (*((Dson::Dict*)(&object)));
	if (dict.contains("preset")) {
		const Dson::String *presetObject = dict["preset"].safeConvert<Dson::String>(errorStream, "Error: DumbBasedShader preset should be of a string type...");
		if (presetObject == NULL) return false;
		const std::string &preset = presetObject->value();
		if (preset == "rough_gold") (*this) = roughGold();
		else if ((preset == "glossy_gold") || (preset == "gold")) (*this) = glossyGold();
		else if ((preset == "glossy_finish") || (preset == "glossy")) (*this) = glossyFinish();
		else if ((preset == "matte_finish") || (preset == "matte")) (*this) = matteFinish();
		else {
			if (errorStream != NULL) (*errorStream) << ("Error: DumbBasedShader preset \"" + preset + "\" does not exist...") << std::endl;
			return false;
		}
	}
	ColorRGB fresnelFactor = fres;
	float metal = specMass;
	float specular = spec;
	ColorRGB diffuse = diff / ((metal == 1.0f) ? 1.0f : (1.0f - metal));
	if (dict.contains("fresnel")) {
		Vector3 fresnelColorVector(0.0f, 0.0f, 0.0f);
		if (!fresnelColorVector.fromDson(dict["fresnel"], errorStream)) return false;
		fresnelFactor = fresnelColorVector;
	}
	if (dict.contains("diffuse")) {
		Vector3 diffuseColorVector(0.0f, 0.0f, 0.0f);
		if (!diffuseColorVector.fromDson(dict["diffuse"], errorStream)) return false;
		diffuse = diffuseColorVector;
	}
	if (dict.contains("specular")) {
		const Dson::Number *specularValue = dict["specular"].safeConvert<Dson::Number>(errorStream, "Error: DumbBasedShader specular value hat to be a number...");
		if (specularValue == NULL) return false;
		specular = specularValue->floatValue();
	}
	if (dict.contains("metal")) {
		const Dson::Number *metalValue = dict["metal"].safeConvert<Dson::Number>(errorStream, "Error: DumbBasedShader metal value hat to be a number...");
		if (metalValue == NULL) return false;
		metal = metalValue->floatValue();
	}
	(*this) = DumbBasedShader(fresnelFactor, specular, diffuse, metal);
	return true;
}
