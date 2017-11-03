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
/** Operators: **/
__device__ __host__ inline Color Color::operator+()const {
	return (*this);
}
__device__ __host__ inline Color Color::operator+(const Color &c)const {
	return Color(r + c.r, g + c.g, b + c.b, a + c.a);
}
__device__ __host__ inline Color& Color::operator+=(const Color &c) {
	SetThisColor(r + c.r, g + c.g, b + c.b, a + c.a);
	return(*this);
}

__device__ __host__ inline Color Color::operator-()const {
	return Color(-r, -g, -b, -a);
}
__device__ __host__ inline Color Color::operator-(const Color &c)const {
	return Color(r - c.r, g - c.g, b - c.b, a - c.a);
}
__device__ __host__ inline Color& Color::operator-=(const Color &c){
	SetThisColor(r - c.r, g - c.g, b - c.b, a - c.a);
	return(*this);
}

__device__ __host__ inline Color Color::operator*(const Color &c)const {
	return Color(r * c.r, g * c.g, b * c.b, a * c.a);
}
__device__ __host__ inline Color Color::operator*(float f)const {
	return Color(r * f, g * f, b * f, a * f);
}
__device__ __host__ inline Color& Color::operator*=(const Color &c) {
	SetThisColor(r * c.r, g * c.g, b * c.b, a * c.a);
	return(*this);
}
__device__ __host__ inline Color& Color::operator*=(float f) {
	SetThisColor(r * f, g * f, b * f, a * f);
	return(*this);
}

__device__ __host__ inline Color Color::operator/(const Color &c)const {
	return Color(r / c.r, g / c.g, b / c.b, a / c.a);
}
__device__ __host__ inline Color Color::operator/(float f)const {
	register float fInv = 1.0f / f;
	return Color(r * fInv, g * fInv, b * fInv, a * fInv);
}
__device__ __host__ inline Color& Color::operator/=(const Color &c) {
	SetThisColor(r / c.r, g / c.g, b / c.b, a / c.a);
	return(*this);
}
__device__ __host__ inline Color& Color::operator/=(float f) {
	register float fInv = 1.0f / f;
	SetThisColor(r * fInv, g * fInv, b * fInv, a * fInv);
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
