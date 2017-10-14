#pragma once
#include "Generic.cuh"
#include "ColorRGB.cuh"
#include "ManagedHandler.cuh"






/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class FrameBufferFunctionPack {
public:
	__dumb__ FrameBufferFunctionPack();
	__dumb__ void clean();
	template<typename BufferType>
	__dumb__ void use();

	__dumb__ int width(const void *data)const;
	__dumb__ int height(const void *data)const;
	__dumb__ Color& color(void *data, int x, int y)const;
	__dumb__ const Color& colorConst(const void *data, int x, int y)const;


private:
	int(*widthFunction)(const void *data);
	int(*heightFunction)(const void *data);
	Color&(*colorFunction)(void *data, int x, int y);
	const Color&(*constColorFunction)(const void *data, int x, int y);
	template<typename DataType>
	__dumb__ static int widthGeneric(const void *data)const;
	template<typename DataType>
	__dumb__ static int heightGeneric(const void *data)const;
	template<typename DataType>
	__dumb__ static Color& colorGeneric(void *data, int x, int y);
	template<typename DataType>
	__dumb__ static const Color& colorConstGeneric(const void *data, int x, int y);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
class FrameBuffer : public Generic<FrameBufferFunctionPack> {
public:
	__dumb__ int width()const;
	__dumb__ int height()const;
	__dumb__ Color& color(int x, int y);
	__dumb__ const Color& color(int x, int y)const;

	inline FrameBuffer *upload()const;
	inline static FrameBuffer* upload(const FrameBuffer *source, int count = 1);
};





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
SPECIALISE_TYPE_TOOLS_FOR(FrameBuffer);





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
typedef ManagedHandler<FrameBuffer> FrameBufferHandler;





#include "FrameBuffer.impl.cuh"

