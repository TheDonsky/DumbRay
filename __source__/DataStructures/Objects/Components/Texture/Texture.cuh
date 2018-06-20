#pragma once
#include "../../../Primitives/Pure/Color/Color.h"
#include "../../../Primitives/Pure/Vector2/Vector2.h"
#include "../../../GeneralPurpose/TypeTools/TypeTools.cuh"



class Texture;
SPECIALISE_TYPE_TOOLS_FOR(Texture);



class Texture {
public:
	enum Filtering {
		FILTER_NONE = 0,
		FILTER_BILINEAR = 1
	};

	__device__ __host__ inline Texture(uint32_t width = 0, uint32_t height = 0, Filtering filtering = FILTER_BILINEAR);
	__device__ __host__ inline Texture(const Texture &other);
	__device__ __host__ inline Texture& operator=(const Texture &other);
	__device__ __host__ inline void copyFrom(const Texture &other);
	__device__ __host__ inline Texture(Texture &&other);
	__device__ __host__ inline Texture& operator=(Texture &&other);
	__device__ __host__ inline void stealFrom(Texture &other);
	__device__ __host__ inline void swapWith(Texture &other);
	__device__ __host__ inline ~Texture();

	__device__ __host__ inline void setReolution(uint32_t width, uint32_t height);
	__device__ __host__ inline void setFiltering(Filtering filter);
	__device__ __host__ inline void clean();

	__device__ __host__ inline uint32_t width()const;
	__device__ __host__ inline uint32_t height()const;
	__device__ __host__ inline Filtering filtering()const;
	__device__ __host__ inline Color* operator[](uint32_t y);
	__device__ __host__ inline const Color* operator[](uint32_t y)const;
	__device__ __host__ inline Color& operator()(uint32_t x, uint32_t y);
	__device__ __host__ inline const Color& operator()(uint32_t x, uint32_t y)const;
	__device__ __host__ inline const Color operator()(const Vector2 &pos)const;



	

	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	DEFINE_CUDA_LOAD_INTERFACE_FOR(Texture);




private:
	Color *data;
	uint32_t w, h;
	Filtering flags;
	Color *protectedData;





	/** ########################################################################## **/
	/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
	/** ########################################################################## **/
	/** Friends: **/
	DEFINE_TYPE_TOOLS_FRIENDSHIP_FOR(Texture);
};



#include "Texture.impl.cuh"
