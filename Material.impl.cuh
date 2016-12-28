#include"Material.cuh"
#include"Shapes.cuh"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
__dumb__ void Shader<HitType>::clean() {
	castFunction = NULL;
}
template<typename HitType>
template<typename ShaderType>
__dumb__ void Shader<HitType>::use() {
	castFunction = castGeneric<ShaderType>;
	bounceFunction = bounceGeneric<ShaderType>;
	illuminateFunction = illuminateGeneric<ShaderType>;
}

template<typename HitType>
__dumb__ ShaderReport Shader<HitType>::cast(const void *shader, const ShaderHitInfo<HitType>& info)const {
	return castFunction(shader, info);
}
template<typename HitType>
__dumb__ void Shader<HitType>::bounce(const void *shader, const ShaderBounceInfo<HitType> &info, ShaderBounce *bounce)const {
	return bounceFunction(shader, info, bounce);
}
template<typename HitType>
__dumb__ Photon Shader<HitType>::illuminate(const void *shader, const ShaderHitInfo<HitType>& info)const {
	return illuminateFunction(shader, info);
}


template<typename HitType>
template<typename ShaderType>
__dumb__ ShaderReport Shader<HitType>::castGeneric(const void *shader, const ShaderHitInfo<HitType>& info) {
	return ((ShaderType*)shader)->cast(info);
}
template<typename HitType>
template<typename ShaderType>
__dumb__  void Shader<HitType>::bounceGeneric(const void *shader, const ShaderBounceInfo<HitType> &info, ShaderBounce *bounce) {
	return ((ShaderType*)shader)->bounce(info, bounce);
}
template<typename HitType>
template<typename ShaderType>
__dumb__ Photon Shader<HitType>::illuminateGeneric(const void *shader, const ShaderHitInfo<HitType>& info)  {
	return ((ShaderType*)shader)->illuminate(info);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
template<typename HitType>
__dumb__ ShaderReport Material<HitType>::cast(const ShaderHitInfo<HitType>& info)const {
	return functions().cast(object(), info);
}
template<typename HitType>
__dumb__ void Material<HitType>::bounce(const ShaderBounceInfo<HitType> &info, ShaderBounce *bounce)const {
	return functions().bounce(object(), info, bounce);
}
template<typename HitType>
__dumb__ Photon Material<HitType>::illuminate(const ShaderHitInfo<HitType>& info)const {
	return functions().illuminate(object(), info);
}

template<typename HitType>
inline Material<HitType>* Material<HitType>::upload()const {
	return ((Material*)Generic<Shader<HitType> >::upload());
}
template<typename HitType>
inline Material<HitType>* Material<HitType>::upload(const Material *source, int count) {
	return ((Material*)Generic<Shader<HitType> >::upload(source, count));
}






/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<typename HitType>
__device__ __host__ inline void TypeTools<Material<HitType> >::init(Material<HitType> &m) {
	TypeTools<Generic<Shader<HitType> > >::init(m);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Material<HitType> >::dispose(Material<HitType> &m) {
	TypeTools<Generic<Shader<HitType> > >::dispose(m);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Material<HitType> >::swap(Material<HitType> &a, Material<HitType> &b) {
	TypeTools<Generic<Shader<HitType> > >::swap(a, b);
}
template<typename HitType>
__device__ __host__ inline void TypeTools<Material<HitType> >::transfer(Material<HitType> &src, Material<HitType> &dst) {
	TypeTools<Generic<Shader<HitType> > >::transfer(src, dst);
}

template<typename HitType>
inline bool TypeTools<Material<HitType> >::prepareForCpyLoad(const Material<HitType> *source, Material<HitType> *hosClone, Material<HitType> *devTarget, int count) {
	return TypeTools<Generic<Shader<HitType> > >::prepareForCpyLoad(source, hosClone, devTarget, count);
}

template<typename HitType>
inline void TypeTools<Material<HitType> >::undoCpyLoadPreparations(const Material<HitType> *source, Material<HitType> *hosClone, Material<HitType> *devTarget, int count) {
	TypeTools<Generic<Shader<HitType> > >::undoCpyLoadPreparations(source, hosClone, devTarget, count);
}

template<typename HitType>
inline bool TypeTools<Material<HitType> >::devArrayNeedsToBeDisposed() {
	return TypeTools<Generic<Shader<HitType> > >::devArrayNeedsToBeDisposed();
}
template<typename HitType>
inline bool TypeTools<Material<HitType> >::disposeDevArray(Material<HitType> *arr, int count) {
	return TypeTools<Generic<Shader<HitType> > >::disposeDevArray(arr, count);
}

