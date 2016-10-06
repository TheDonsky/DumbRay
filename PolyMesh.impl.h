#include"PolyMesh.h"





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__device__ __host__ inline PolyMesh::IndexNode::IndexNode(){ }
__device__ __host__ inline PolyMesh::IndexNode::IndexNode(int v, int n, int t){
	vert = v;
	norm = n;
	tex = t;
}

__device__ __host__ inline PolyMesh::BakedNode::BakedNode(){ }
__device__ __host__ inline PolyMesh::BakedNode::BakedNode(Vertex v, Vector3 n, Vertex t){
	vert = v;
	norm = n;
	tex = t;
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__device__ __host__ inline PolyMesh::PolyMesh(){ }
__device__ __host__ inline PolyMesh::PolyMesh(const VertexList &vertices, const VertexList &normals, const VertexList &textures, const FaceList &indexFaces){
	constructFrom(vertices, normals, textures, indexFaces);
}
__device__ __host__ inline PolyMesh::PolyMesh(const PolyMesh &m){
	(*this) = m;
}
__device__ __host__ inline PolyMesh& PolyMesh::operator=(const PolyMesh &m){
	return(constructFrom(m.data.verts, m.data.norms, m.data.texs, m.faces));
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

__device__ __host__ inline int PolyMesh::vertextCount()const{
	return(data.verts.size());
}
__device__ __host__ inline Vertex& PolyMesh::vertex(int index){
	return(data.verts[index]);
}
__device__ __host__ inline const Vertex& PolyMesh::vertex(int index)const{
	return(data.verts[index]);
}
__device__ __host__ inline void PolyMesh::addVertex(const Vertex &v){
	data.verts.push(v);
}
__device__ __host__ inline void PolyMesh::removeVertex(int index){
	removeVert(data.verts, index, faces, getVert, setVert);
}

__device__ __host__ inline int PolyMesh::normalCount()const{
	return(data.norms.size());
}
__device__ __host__ inline Vector3& PolyMesh::normal(int index){
	return(data.norms[index]);
}
__device__ __host__ inline const Vector3& PolyMesh::normal(int index)const{
	return(data.norms[index]);
}
__device__ __host__ inline void PolyMesh::addNormal(const Vector3 &n){
	data.norms.push(n);
}
__device__ __host__ inline void PolyMesh::removeNormal(int index){
	removeVert(data.norms, index, faces, getNorm, setNorm);
}

__device__ __host__ inline int PolyMesh::textureCount()const{
	return(data.texs.size());
}
__device__ __host__ inline Vertex& PolyMesh::texture(int index){
	return(data.texs[index]);
}
__device__ __host__ inline const Vertex& PolyMesh::texture(int index)const{
	return(data.texs[index]);
}
__device__ __host__ inline void PolyMesh::addTexture(const Vertex &t){
	data.texs.push(t);
}
__device__ __host__ inline void PolyMesh::removeTexture(int index){
	removeVert(data.texs, index, faces, getTex, setTex);
}

__device__ __host__ inline int PolyMesh::faceCount()const{
	return(faces.size());
}
__device__ __host__ inline PolyMesh::IndexFace& PolyMesh::indexFace(int index){
	return(faces[index]);
}
__device__ __host__ inline const PolyMesh::Face PolyMesh::face(int index)const{
	Face face;
	const IndexFace &f = faces[index];
	for (int ind = 0; ind < f.size(); ind++){
		const IndexNode &nd = f[ind];
		face.push(BakedNode(data.verts[nd.vert], data.norms[nd.norm], data.texs[nd.tex]));
	}
	return(face);
}
__device__ __host__ inline const PolyMesh::IndexFace& PolyMesh::indexFace(int index)const{
	return(faces[index]);
}
__device__ __host__ inline void PolyMesh::addFace(const IndexFace &f){
	faces.push(f);
}
__device__ __host__ inline void PolyMesh::addFace(const Face &f){
	IndexFace face;
	for (int i = 0; i < f.size(); i++){
		face.push(IndexNode(data.verts.size(), data.norms.size(), data.texs.size()));
		addVertex(f[i].vert);
		addNormal(f[i].norm);
		addTexture(f[i].tex);
	}
	addFace(face);
}
__device__ __host__ inline void PolyMesh::removeFace(int index){
	faces.remove(index);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__device__ __host__ inline void PolyMesh::bake(BakedTriMesh &mesh)const{
	for (int i = 0; i < faceCount(); i++){
		const Face f = face(i);
		int prev = 1;
		for (int cur = 2; cur < f.size(); cur++){
			register Triangle vert(f[0].vert, f[prev].vert, f[cur].vert);
			register Triangle norm(f[0].norm, f[prev].norm, f[cur].norm);
			register Triangle tex(f[0].tex, f[prev].tex, f[cur].tex);
			mesh.push(BakedTriFace(vert, norm, tex));
			prev = cur;
		}
	}
}
__device__ __host__ inline BakedTriMesh PolyMesh::bake()const{
	BakedTriMesh mesh;
	bake(mesh);
	return(mesh);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
__device__ __host__ inline PolyMesh& PolyMesh::constructFrom(const VertexList &vertices, const VertexList &normals, const VertexList &textures, const FaceList &indexFaces){
	data.verts = vertices;
	data.norms = normals;
	data.texs = textures;
	faces = indexFaces;
	return(*this);
}

__device__ __host__ inline int PolyMesh::getVert(IndexNode &node){
	return(node.vert);
}
__device__ __host__ inline void PolyMesh::setVert(IndexNode &node, int value){
	node.vert = value;
}
__device__ __host__ inline int PolyMesh::getNorm(IndexNode &node){
	return(node.norm);
}
__device__ __host__ inline void PolyMesh::setNorm(IndexNode &node, int value){
	node.norm = value;
}
__device__ __host__ inline int PolyMesh::getTex(IndexNode &node){
	return(node.tex);
}
__device__ __host__ inline void PolyMesh::setTex(IndexNode &node, int value){
	node.tex = value;
}
__device__ __host__ inline void PolyMesh::removeVert(VertexList &vertData, int index, FaceList &faceBuffer, int(*indexGet)(IndexNode&), void(*indexSet)(IndexNode&, int)){
	for (int i = 0; i < faceBuffer.size(); i++){
		for (int j = 0; j < faceBuffer[i].size(); j++){
			if (indexGet(faceBuffer[i][j]) == index){
				faceBuffer[i].remove(j);
				j--;
			}
			else if (indexGet(faceBuffer[i][j]) == (vertData.size() - 1))
				indexSet(faceBuffer[i][j], index);
		}
		if (faceBuffer[i].size() < 3){
			faceBuffer.swapPop(i);
			i--;
		}
	}
	vertData.remove(index);
}





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
IMPLEMENT_CUDA_LOAD_INTERFACE_FOR(PolyMesh);





/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/
/** Friends: **/
template<>
__device__ __host__ inline void StacktorTypeTools<PolyMesh>::init(PolyMesh &m){
	StacktorTypeTools<PolyMesh::VertexList>::init(m.data.verts);
	StacktorTypeTools<PolyMesh::VertexList>::init(m.data.norms);
	StacktorTypeTools<PolyMesh::VertexList>::init(m.data.texs);
	StacktorTypeTools<PolyMesh::FaceList>::init(m.faces);
}
template<>
__device__ __host__ inline void StacktorTypeTools<PolyMesh>::dispose(PolyMesh &m){
	StacktorTypeTools<PolyMesh::VertexList>::dispose(m.data.verts);
	StacktorTypeTools<PolyMesh::VertexList>::dispose(m.data.norms);
	StacktorTypeTools<PolyMesh::VertexList>::dispose(m.data.texs);
	StacktorTypeTools<PolyMesh::FaceList>::dispose(m.faces);
}
template<>
__device__ __host__ inline void StacktorTypeTools<PolyMesh>::swap(PolyMesh &a, PolyMesh &b){
	StacktorTypeTools<PolyMesh::VertexList>::swap(a.data.verts, b.data.verts);
	StacktorTypeTools<PolyMesh::VertexList>::swap(a.data.norms, b.data.norms);
	StacktorTypeTools<PolyMesh::VertexList>::swap(a.data.texs, b.data.texs);
	StacktorTypeTools<PolyMesh::FaceList>::swap(a.faces, b.faces);
}
template<>
__device__ __host__ inline void StacktorTypeTools<PolyMesh>::transfer(PolyMesh &src, PolyMesh &dst){
	StacktorTypeTools<PolyMesh::VertexList>::transfer(src.data.verts, dst.data.verts);
	StacktorTypeTools<PolyMesh::VertexList>::transfer(src.data.norms, dst.data.norms);
	StacktorTypeTools<PolyMesh::VertexList>::transfer(src.data.texs, dst.data.texs);
	StacktorTypeTools<PolyMesh::FaceList>::transfer(src.faces, dst.faces);
}

template<>
inline bool StacktorTypeTools<PolyMesh>::prepareForCpyLoad(const PolyMesh *source, PolyMesh *hosClone, PolyMesh *devTarget, int count){
	int i = 0;
	for (i = 0; i < count; i++){
		PolyMesh::VertexList *srcData = (PolyMesh::VertexList*)(&(source[i].data));
		PolyMesh::VertexList *hosData = (PolyMesh::VertexList*)(&(hosClone[i].data));
		PolyMesh::VertexList *devData = (PolyMesh::VertexList*)(&((devTarget + i)->data));
		if (!StacktorTypeTools<PolyMesh::VertexList>::prepareForCpyLoad(srcData, hosData, devData, 3)) break;
		if (!StacktorTypeTools<PolyMesh::FaceList>::prepareForCpyLoad(&(source[i].faces), &(hosClone[i].faces), &((devTarget + i)->faces), 1)){
			StacktorTypeTools<PolyMesh::VertexList>::undoCpyLoadPreparations(srcData, hosData, devData, 3);
			break;
		}
	}
	if (i < count){
		undoCpyLoadPreparations(source, hosClone, devTarget, i);
		return(false);
	}
	return(true);
}
template<>
inline void StacktorTypeTools<PolyMesh>::undoCpyLoadPreparations(const PolyMesh *source, PolyMesh *hosClone, PolyMesh *devTarget, int count){
	for (int i = 0; i < count; i++){
		PolyMesh::VertexList *srcData = (PolyMesh::VertexList*)(&(source[i].data));
		PolyMesh::VertexList *hosData = (PolyMesh::VertexList*)(&(hosClone[i].data));
		PolyMesh::VertexList *devData = (PolyMesh::VertexList*)(&((devTarget + i)->data));
		StacktorTypeTools<PolyMesh::VertexList>::undoCpyLoadPreparations(srcData, hosData, devData, 3);
		StacktorTypeTools<PolyMesh::FaceList>::undoCpyLoadPreparations(&(source[i].faces), &(hosClone[i].faces), &((devTarget + i)->faces), 1);
	}
}
template<>
inline bool StacktorTypeTools<PolyMesh>::devArrayNeedsToBeDisoposed(){
	return(StacktorTypeTools<PolyMesh::VertexList>::devArrayNeedsToBeDisoposed() || StacktorTypeTools<PolyMesh::FaceList>::devArrayNeedsToBeDisoposed());
}
template<>
inline bool StacktorTypeTools<PolyMesh>::disposeDevArray(PolyMesh *arr, int count){
	for (int i = 0; i < count; i++){
		if (!StacktorTypeTools<PolyMesh::VertexList>::disposeDevArray((PolyMesh::VertexList*)(&((arr + i)->data)), 3)) return(false);
		if (!StacktorTypeTools<PolyMesh::FaceList>::disposeDevArray(&((arr + i)->faces), 1)) return(false);
	}
	return(true);
}
