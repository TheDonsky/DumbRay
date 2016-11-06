#include"Photon.cuh"




__dumb__ Photon::Photon(){}
__dumb__ Photon::Photon(const Ray &r, const ColorRGB &c){
	ray = r;
	color = c;
}




__device__ __host__ inline Photon Photon::operator>>(const Transform &trans)const {
	return Photon((ray >> trans), color);
}
__device__ __host__ inline Photon& Photon::operator>>=(const Transform &trans) {
	ray >>= trans;
	return (*this);
}
__device__ __host__ inline Photon Photon::operator<<(const Transform &trans)const {
	return Photon((ray << trans), color);
}
__device__ __host__ inline Photon& Photon::operator<<=(const Transform &trans) {
	ray <<= trans;
	return (*this);
}
