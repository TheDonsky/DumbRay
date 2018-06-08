#include "SimpleStochasticLense.cuh"


__dumb__ SimpleStochasticLense::SimpleStochasticLense(float angle) {
	if (angle < 1) angle = 1;
	else if (angle > 179) angle = 179;
	register float radian = (angle / (2 * RADIAN));
	x = 0.5f / tan(radian);
}

__dumb__ void SimpleStochasticLense::getPixelSamples(const LenseGetPixelSamplesRequest &request, RaySamples *samples)const {
	samples->sampleCount = 1;
	float off = (request.pixelSize / 2.0f);
	DumbRand *dRand = request.context->entropy;
	samples->samples[0] = SampleRay(
		Ray(Vector3(0.0f, 0.0f, 0.0f), 
			Vector3(
				request.screenSpacePosition.x + dRand->range(-off, off), 
				request.screenSpacePosition.y + dRand->range(-off, off), x)),
		1.0f);
}
__dumb__ Color SimpleStochasticLense::getPixelColor(const LenseGetPixelColorRequest &request)const {
	if (request.photonType == PHOTON_TYPE_DIRECT_ILLUMINATION) return Color(0.0f, 0.0f, 0.0f, 0.0f);
	else return request.photon.color;
}


inline bool SimpleStochasticLense::fromDson(const Dson::Object &object, std::ostream *errorStream) {
	if (object.type() != Dson::Object::DSON_DICT) {
		if (errorStream != NULL) (*errorStream) << "Error: SimpleStochasticLense can only accept Dson::Dict in fromDson method..." << std::endl;
		return false;
	}
	const Dson::Dict &dict = (*((Dson::Dict*)(&object)));
	if (dict.contains("angle")) {
		const Dson::Object &angleObject = dict["angle"];
		if (angleObject.type() != Dson::Object::DSON_NUMBER) {
			if (errorStream != NULL) (*errorStream) << "Error: SimpleStochasticLense angle can only be a number..." << std::endl;
			return false;
		}
		(*this) = SimpleStochasticLense(((Dson::Number*)(&angleObject))->floatValue());
	}
	return true;
}

