#include"Photon.cuh"




__dumb__ Photon::Photon(){}
__dumb__ Photon::Photon(const Ray &r, const ColorRGB &c){
	ray = r;
	color = c;
}
