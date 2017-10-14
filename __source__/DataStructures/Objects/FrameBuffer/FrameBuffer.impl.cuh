#include "FrameBuffer.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ FrameBufferFunctionPack::FrameBufferFunctionPack() {
	clean();
}
__dumb__ void FrameBufferFunctionPack::clean() {
	widthFunction = NULL;
	heightFunction = NULL;
	colorFunction = NULL;
	constColorFunction = NULL;
}
template<typename BufferType>
__dumb__ void FrameBufferFunctionPack::use() {
	widthFunction = widthGeneric<BufferType>;
	heightFunction = heightGeneric<BufferType>;
	colorFunction = colorGeneric<BufferType>;
	constColorFunction = colorConstGeneric<BufferType>;
}

__dumb__ int FrameBufferFunctionPack::width(const void *data)const {
	return widthFunction(data);
}
__dumb__ int FrameBufferFunctionPack::height(const void *data)const {
	return heightFunction(data);
}
__dumb__ Color& FrameBufferFunctionPack::color(void *data, int x, int y)const {
	return colorFunction(data, x, y);
}
__dumb__ const Color& FrameBufferFunctionPack::colorConst(const void *data, int x, int y)const {
	return constColorFunction(data, x, y);
}




template<typename DataType>
__dumb__ int FrameBufferFunctionPack::widthGeneric(const void *data)const {
	return ((DataType*)data)->width();
}
template<typename DataType>
__dumb__ int FrameBufferFunctionPack::heightGeneric(const void *data)const {
	return ((DataType*)data)->height();
}
template<typename DataType>
__dumb__ Color& FrameBufferFunctionPack::colorGeneric(void *data, int x, int y) {
	return ((DataType*)data)->color(x, y);
}
template<typename DataType>
__dumb__ const Color& FrameBufferFunctionPack::colorConstGeneric(const void *data, int x, int y) {
	return ((const DataType*)data)->color(x, y);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__dumb__ int FrameBuffer::width()const {
	return functions().width(object());
}
__dumb__ int FrameBuffer::height()const {
	return functions().height(object());
}
__dumb__ Color& FrameBuffer::color(int x, int y) {
	return functions().color(object(), x, y);
}
__dumb__ const Color& FrameBuffer::color(int x, int y)const {
	return functions().colorConst(object(), x, y);
}

inline FrameBuffer* FrameBuffer::upload()const {
	return ((FrameBuffer*)Generic<FrameBufferFunctionPack>::upload());
}
inline FrameBuffer* FrameBuffer::upload(const FrameBuffer *source, int count) {
	return ((FrameBuffer*)Generic<FrameBufferFunctionPack>::upload(source, count));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
COPY_TYPE_TOOLS_IMPLEMENTATION(FrameBuffer, Generic<FrameBufferFunctionPack>);




