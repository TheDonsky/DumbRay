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



__dumb__ float Photon::energy()const {
	return (color.r + color.b + color.g);
}
__dumb__ float Photon::dead()const {
	return (energy() < 0.0001f);
}



__dumb__ Photon Photon::zero() {
	return Photon(Ray(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 0.0f)), ColorRGB(0.0f, 0.0f, 0.0f));
}

