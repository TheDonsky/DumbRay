#include"ColorRGB.cuh"


/* -------------------------------------------------------------------------- */
// Macros:
#define SetThisColorRGB(R, G, B) r = R; g = G; b = B




/** -------------------------------------------------------------------------- **/
/** Construction: **/
__device__ __host__ inline ColorRGB::ColorRGB() {}
__device__ __host__ inline ColorRGB::ColorRGB(float R, float G, float B) {
	SetThisColorRGB(R, G, B);
}
__device__ __host__ inline ColorRGB::ColorRGB(const ColorRGB &c) {
	SetThisColorRGB(c.r, c.g, c.b);
}
__device__ __host__ inline ColorRGB& ColorRGB::operator()(float R, float G, float B) {
	SetThisColorRGB(R, G, B);
	return(*this);
}





/** -------------------------------------------------------------------------- **/
/** Casts: **/
__dumb__ ColorRGB::operator Color()const {
	return Color(r, g, b, 1);
}
__dumb__ ColorRGB::operator Vector3()const {
	return Vector3(r, g, b);
}
__dumb__ ColorRGB::ColorRGB(const Color &c) {
	SetThisColorRGB(c.r, c.g, c.b);
}
__dumb__ ColorRGB::ColorRGB(const Vector3 &v) {
	SetThisColorRGB(v.x, v.y, v.z);
}





/** -------------------------------------------------------------------------- **/
/** Stream operators: **/
std::istream& operator >> (std::istream &stream, ColorRGB &c) {
	return(stream >> c.r >> c.g >> c.b);
}
std::ostream& operator<<(std::ostream &stream, const ColorRGB &c) {
	stream << "(Red: " << (int)(c.r * 1000) / 1000.0 << "; Green: " << (int)(c.g * 1000) / 1000.0;
	return(stream << "; Blue: " << (int)(c.b * 1000) / 1000.0 << ")");
}





/** -------------------------------------------------------------------------- **/
/** Macro undefs: **/
#undef SetThisColorRGB

