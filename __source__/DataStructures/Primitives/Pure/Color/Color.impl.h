#include"Color.h"


/* -------------------------------------------------------------------------- */
// Macros:
#define SetThisColor(R, G, B, A) r = R; g = G; b = B; a = A




/** -------------------------------------------------------------------------- **/
/** Construction: **/
__device__ __host__ inline Color::Color(){}
__device__ __host__ inline Color::Color(float R, float G, float B, float A){
	SetThisColor(R, G, B, A);
}
__device__ __host__ inline Color::Color(const Color &c){
	SetThisColor(c.r, c.g, c.b, c.a);
}
__device__ __host__ inline Color& Color::operator()(float R, float G, float B, float A){
	SetThisColor(R, G, B, A);
	return(*this);
}





/** -------------------------------------------------------------------------- **/
/** Stream operators: **/
inline static std::istream& operator>>(std::istream &stream, Color &c){
	return(stream >> c.r >> c.g >> c.b >> c.a);
}
inline static std::ostream& operator<<(std::ostream &stream, const Color &c){
	stream << "(Red: " << (int)(c.r * 1000) / 1000.0 << "; Green: " << (int)(c.g * 1000) / 1000.0;
	return(stream << "; Blue: " << (int)(c.b * 1000) / 1000.0 << "; Alpha: " << (int)(c.a * 1000) / 1000.0 << ")");
}





/** -------------------------------------------------------------------------- **/
/** Macro undefs: **/
#undef SetThisColor
