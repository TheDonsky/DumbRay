#include"MeshReader.h"
#include<stdlib.h>
#include"../Tests/Tests.h"

/** ########################################################################## **/
/** //\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\//\\// **/
/** ########################################################################## **/

namespace MeshReaderPrivate{
	inline static bool isFrom(const char *breakpoints, char c){
		int i = 0;
		while (breakpoints[i] != '\0'){
			if (c == breakpoints[i]) return true;
			i++;
		}
		return false;
	}

	inline static bool moveToRelevant(const char *file, int &cursor, const char *breakpoints = "", const char *instantBreaks = "", const char *lineBreaks = ""){
		while (file[cursor] != '\0' && (iswspace(file[cursor]) || isFrom(breakpoints, file[cursor]))){
			if (isFrom(instantBreaks, file[cursor])) return false;
			cursor++;
		}
		return (!(isFrom(lineBreaks, file[cursor])));
	}

	inline static bool getLine(const char *file, int &cursor, String &dst){
		dst.clear();
		if (moveToRelevant(file, cursor))
			while (file[cursor] != '\n' && file[cursor] != '\0'){
				dst.push(file[cursor]);
				cursor++;
			}
		while (dst.size() > 0 && iswspace(dst.top())) dst.pop();
		dst.push('\0');
		return(dst.size() > 1);
	}

	inline static bool getToken(const char *file, int &cursor, String &dst, const char *breakpoints = "", const char *instantBreaks = "", const char *lineBreaks = "#"){
		dst.clear();
		if (moveToRelevant(file, cursor, breakpoints, instantBreaks, lineBreaks)){
			while (file[cursor] != '\0' && (!(iswspace(file[cursor]) || isFrom(breakpoints, file[cursor])))){
				dst.push(file[cursor]);
				cursor++;
			}
			dst.push('\0');
		}
		else if (file[cursor] != '\0' && isFrom(lineBreaks, file[cursor])){
			getLine(file, cursor, dst);
			return false;
		}
		else dst.push('\0');
		return(dst.size() > 1);
	}

	inline static bool equals(const String &s, const char *str){
		for (int i = 0; i < s.size(); i++)
			if (s[i] != str[i]) return(false);
		return(true);
	}

	inline static float getFloat(const char *file, int &cursor, bool fromThisLine = false){
		String token;
		if (fromThisLine){ if (!getToken(file, cursor, token, "", "\n")) return 0; }
		else if (!getToken(file, cursor, token)) return 0;
		return (float)atof(token + 0);
	}

	inline static Vertex getVertex(const char *file, int &cursor){
		float x = getFloat(file, cursor);
		float y = getFloat(file, cursor);
		float z = getFloat(file, cursor);
		return(Vertex(x, y, z));
	}
	
	inline static Vertex getVertexFromThisLine(const char *file, int &cursor){
		float x = getFloat(file, cursor, true);
		float y = getFloat(file, cursor, true);
		float z = getFloat(file, cursor, true);
		return(Vertex(x, y, z));
	}

	inline static int getInt(const String &token, int &cursor, const char *breakpoints){
		String dig;
		while (cursor < token.size() && (!isFrom(breakpoints, token[cursor]))){
			dig.push(token[cursor]);
			cursor++;
		}
		dig.push('\0');
		if (dig.size() > 1) return atoi(dig + 0);
		else return 0;
	}

	inline static PolyMesh::IndexNode getNode(const String &token, const char *spacers = "/\0"){
		PolyMesh::IndexNode node(0, 0, 0);
		int cursor = 0;
		node.vert = getInt(token, cursor, spacers) - 1; cursor++;
		node.tex = getInt(token, cursor, spacers) - 1; cursor++;
		node.norm = getInt(token, cursor, spacers) - 1;
		return(node);
	}

	inline static void getIndexFace(const char *file, int &cursor, PolyMesh::IndexFace &face){
		String token;
		while (getToken(file, cursor, token, "", "\n"))
			face.push(getNode(token));
	}

	inline static void printIndexNode(const PolyMesh::IndexNode &node, const char *start, const char *mid, const char *end, std::ostream &o = std::cout){
		o << start << node.vert << mid << node.tex << mid << node.norm << end;
	}

	inline static void printIndexFace(const PolyMesh::IndexFace &face, const char*start, const char *mid, const char *end, const char *nodeStart, const char *nodeMid, const char *nodeEnd, std::ostream &o = std::cout){
		o << start;
		for (int i = 0; i < face.size(); i++){
			printIndexNode(face[i], nodeStart, nodeMid, nodeEnd, o);
			if (i < (face.size() - 1))
				o << mid;
		}
		o << end;
	}

	inline static void printIndexNode(const PolyMesh::IndexNode &node){
		printIndexNode(node, "(", ", ", ")");
	}

	inline static void printIndexFace(const PolyMesh::IndexFace &face){
		printIndexFace(face, "[", ", ", "]", "(", ", ", ")");
	}

	inline static void printIndexNode(const PolyMesh::IndexNode &node, std::ostream &o){
		printIndexNode(node, "", "/", "/", o);
	}

	inline static void printIndexFace(const PolyMesh::IndexFace &face, std::ostream &o){
		printIndexFace(face, "", " ", "", "", "/", "", o);
	}

	inline static void addVertex(PolyMesh::VertexList &collection, const char *file, int &cursor, bool dump, const char *comment, const Vector3 &scale){
		collection.push(getVertexFromThisLine(file, cursor) ^ scale);
		if (dump) std::cout << comment << collection.top() << std::endl;
	}

	inline static void addFace(PolyMesh::FaceList &faces, const char *file, int &cursor, bool dump){
		faces.push(PolyMesh::IndexFace());
		getIndexFace(file, cursor, faces.top());
		if (dump){
			std::cout << "FACE: ";
			printIndexFace(faces.top());
			std::cout << std::endl;
		}
	}

	inline static void addObject(
			const Stacktor<PolyMesh> &meshList,
			Stacktor<String> &nameList, 
			const PolyMesh::FaceList &faces, 
			Stacktor<Pair<int, int> > &objects,
			int &start, 
			const char *file, 
			int &cursor, 
			bool dump){

		if (faces.size() > start){
			objects.push({ start, faces.size() });
			start = faces.size();
			if (nameList.size() < meshList.size()) nameList.push("NAMELESS OBJECT");
		}
		else while (nameList.size() > meshList.size()) nameList.pop();
		nameList.push(String());
		getLine(file, cursor, nameList.top());
		if (dump) std::cout << "OBJECT NAME: " << (nameList.top() + 0) << std::endl;
	}

	inline static void read(
			const Stacktor<PolyMesh> &meshList,
			Stacktor<String> &nameList, 
			PolyMesh::VertexList &verts, 
			PolyMesh::VertexList &norms, 
			PolyMesh::VertexList &texs, 
			PolyMesh::FaceList &faces, 
			Stacktor<Pair<int, int> > &objects, 
			const char *file, 
			bool dump){

		int start = 0;
		int cursor = 0;
		while (file[cursor] != '\0'){
			String s;
			if (getToken(file, cursor, s)){
				if (equals(s, "v")) addVertex(verts, file, cursor, dump, "VERTEX: ", Vector3(-1, 1, 1));
				else if (equals(s, "vt")) addVertex(texs, file, cursor, dump, "TEXTURE: ", Vector3(1, 1, 1));
				else if (equals(s, "vn")) addVertex(norms, file, cursor, dump, "NORMAL: ", Vector3(-1, 1, 1));
				else if (equals(s, "f")) addFace(faces, file, cursor, dump);
				else if (equals(s, "g")) addObject(meshList, nameList, faces, objects, start, file, cursor, dump);
				else if (dump) std::cout << "IGNORED TOKEN: " << (s + 0) << std::endl;
			}
			else if (dump) std::cout << "IGNORED LINE: " << (s + 0) << std::endl;
		}

		if (faces.size() > start){
			objects.push({ start, faces.size() });
			if (nameList.size() < meshList.size()) nameList.push("NAMELESS OBJECT");
		}
		else while (nameList.size() > meshList.size()) nameList.pop();

		if (dump){
			std::cout << std::endl << "############### OBJECT(S): ###############" << std::endl;
			for (int i = 0; i < objects.size(); i++){
				std::cout << (nameList[i] + 0) << " - faces [" << objects[i].first << " - " << objects[i].second << "]" << std::endl;
			}
		}
	}

	inline static bool createVertexNormals(
			PolyMesh::VertexList &vertexNormals, 
			const PolyMesh::VertexList &verts, 
			const PolyMesh::FaceList &faces, bool 
			dump){

		for (int i = 0; i < verts.size(); i++)
			vertexNormals.push(Vertex::zero());
		for (int i = 0; i < faces.size(); i++)
			for (int j = 0; j < faces[i].size(); j++){
				int prev = ((j - 1 + faces[i].size()) % faces[i].size());
				int next = ((j + 1) % faces[i].size());
				bool nodeValid = (faces[i][prev].vert >= 0 && faces[i][prev].vert < verts.size()) && (faces[i][next].vert >= 0 && faces[i][next].vert < verts.size());
				if (nodeValid) vertexNormals[faces[i][j].vert] += ((verts[faces[i][prev].vert] - verts[faces[i][j].vert]) & (verts[faces[i][next].vert] - verts[faces[i][j].vert]));
				else{
					if (dump){
						std::cout << "ERROR: face";
						printIndexFace(faces[i]);
						std::cout << std::endl;
					}
					return false;
				}
			}
		for (int i = 0; i < verts.size(); i++)
			if (vertexNormals[i] != Vertex::zero())
				vertexNormals[i].normalize();
		return true;
	}

	struct ExtructObjectParamPack {
		PolyMesh *meshAddr;
		const PolyMesh::VertexList *vertsAddr;
		const PolyMesh::VertexList *vertexNormalsAddr; 
		const PolyMesh::VertexList *normsAddr; 
		const PolyMesh::VertexList *texsAddr; 
		const PolyMesh::FaceList *facesAddr; 
		int start, end; 
		volatile bool *passed;
		bool dump;
	};

	inline static void extructObject(ExtructObjectParamPack pack){
		

		PolyMesh &mesh = (*pack.meshAddr);
		const PolyMesh::VertexList &verts = (*pack.vertsAddr);
		const PolyMesh::VertexList &vertexNormals = (*pack.vertexNormalsAddr);
		const PolyMesh::VertexList &norms = (*pack.normsAddr);
		const PolyMesh::VertexList &texs = (*pack.texsAddr);
		const PolyMesh::FaceList &faces = (*pack.facesAddr);
		
		IntMap<int> vertIndexes;
		IntMap<int> normIndexes;
		IntMap<int> texIndexes;

		for (int i = pack.start; i < pack.end; i++){
			mesh.addFace(PolyMesh::IndexFace());
			PolyMesh::IndexFace &face = mesh.indexFace(mesh.faceCount() - 1);
			for (int j = 0; j < faces[i].size(); j++){
				if (!vertIndexes.contains(faces[i][j].vert)){
					if (faces[i][j].vert < 0 || faces[i][j].vert >= verts.size()){
						(*pack.passed) = false;
						if(pack.dump) std::cout << "VERTEX ERROR" << std::endl;
						return;
					}
					vertIndexes[faces[i][j].vert] = mesh.vertextCount();
					mesh.addVertex(verts[faces[i][j].vert]);
				}
				if (!normIndexes.contains(faces[i][j].norm)){
					if (faces[i][j].norm >= norms.size()){
						(*pack.passed) = false;
						if (pack.dump) std::cout << "NORMAL ERROR" << std::endl;
						return;
					}
					normIndexes[faces[i][j].norm] = mesh.normalCount();
					if (faces[i][j].norm >= 0)
						mesh.addNormal(norms[faces[i][j].norm]);
					else mesh.addNormal(vertexNormals[faces[i][j].vert]);
				}
				if (!texIndexes.contains(faces[i][j].tex)){
					if (faces[i][j].tex >= texs.size()){
						(*pack.passed) = false;
						if (pack.dump) std::cout << "TEXTURE ERROR" << std::endl;
						return;
					}
					texIndexes[faces[i][j].tex] = mesh.textureCount();
					if (faces[i][j].tex >= 0)
						mesh.addTexture(texs[faces[i][j].tex]);
					else mesh.addTexture(Vertex::zero());
				}
				face.push(PolyMesh::IndexNode(vertIndexes[faces[i][j].vert], normIndexes[faces[i][j].norm], texIndexes[faces[i][j].tex]));
			}
		}
	}

	inline static bool extructObjects(
			Stacktor<PolyMesh> &meshList, 
			const PolyMesh::VertexList &verts, 
			const PolyMesh::VertexList &norms, 
			const PolyMesh::VertexList &texs, 
			const PolyMesh::FaceList &faces, 
			const Stacktor<Pair<int, int> > &objects,
			bool dump){

		if (objects.size() <= 0) return true;

		PolyMesh::VertexList vertexNormals;
		if (dump) Tests::logLine();
		if (!createVertexNormals(vertexNormals, verts, faces, dump)){
			if(dump) std::cout << "UNABLE TO CALCULATE VERTEX NORMALS" << std::endl;
			return false;
		}
		volatile bool passed = true;
		int start = meshList.size();
		meshList.flush(objects.size());
		std::thread *threads = new std::thread[objects.size()];
		for (int i = 0; i < objects.size(); i++)
			threads[i] = std::thread(extructObject, 
				ExtructObjectParamPack{meshList + i + start, &verts, &vertexNormals, &norms, &texs, &faces, objects[i].first, objects[i].second, &passed, dump});
		for (int i = 0; i < objects.size(); i++)
			threads[i].join();
		delete[] threads;
		return passed;
	}
}








inline static bool MeshReader::readObj(Stacktor<PolyMesh> &meshList, Stacktor<String> &nameList, std::string filename, bool dump){
	bool passed = true;
	std::ifstream fileStream(filename);
	if (fileStream.fail()) {
		if (dump) std::cout << "ERROR opening file: " << filename << std::endl;
		return false;
	}
	std::string file((std::istreambuf_iterator<char>(fileStream)), std::istreambuf_iterator<char>());
	std::string next; fileStream >> next;
	if (!fileStream.eof()) passed = false;
	fileStream.close();
	if (!passed) return false;
	
	PolyMesh::VertexList verts;
	PolyMesh::VertexList norms;
	PolyMesh::VertexList texs;
	PolyMesh::FaceList faces;
	Stacktor<Pair<int, int> > objects;

	MeshReaderPrivate::read(meshList, nameList, verts, norms, texs, faces, objects, file.c_str(), dump);
	return MeshReaderPrivate::extructObjects(meshList, verts, norms, texs, faces, objects, dump);
}


inline static bool MeshReader::writeObj(const Stacktor<PolyMesh> &meshList, const Stacktor<String> &nameList, std::string filename) {
	std::ofstream stream(filename);
	if (stream.fail()) return false;
	stream << std::fixed;
	stream << "# DUMPED WITH DumbRay .obj EXPORTER" << std::endl;
	int vertsSoFar = 1;
	int normsSoFar = 1;
	int texsSoFar = 1;
	for (int i = 0; i < meshList.size(); i++) {
		stream << std::endl << std::endl << std::endl;
		std::string objectName = ((i >= nameList.size()) ? (std::string("Object ") + std::to_string(i)) : (nameList[i] + 0));
		stream << "#######################################################" << std::endl;
		stream << "# OBJECT: " << objectName << std::endl << std::endl;
		stream << "# -----------------------------" << std::endl;
		for (int j = 0; j < meshList[i].vertextCount(); j++)
			stream << "v " << (-meshList[i].vertex(j).x) << " " << meshList[i].vertex(j).y << " " << meshList[i].vertex(j).z << std::endl;
		stream << "# " << meshList[i].vertextCount() << " VERTICES" << std::endl << std::endl;
		stream << "# -----------------------------" << std::endl;
		for (int j = 0; j < meshList[i].normalCount(); j++)
			stream << "vn " << (-meshList[i].normal(j).x) << " " << meshList[i].normal(j).y << " " << meshList[i].normal(j).z << std::endl;
		stream << "# " << meshList[i].normalCount() << " NORMALS" << std::endl << std::endl;
		stream << "# -----------------------------" << std::endl;
		for (int j = 0; j < meshList[i].textureCount(); j++)
			stream << "vt " << meshList[i].texture(j).x << " " << meshList[i].texture(j).y << " " << meshList[i].texture(j).z << std::endl;
		stream << "# " << meshList[i].textureCount() << " TEXTURE COORDINATES" << std::endl << std::endl;
		stream << "# -----------------------------" << std::endl;
		stream << "g " << objectName << std::endl;
		for (int j = 0; j < meshList[i].faceCount(); j++) {
			stream << "f";
			for (int k = 0; k < meshList[i].indexFace(j).size(); k++) {
				bool vertHere = (meshList[i].indexFace(j)[k].vert >= 0);
				bool normHere = (meshList[i].indexFace(j)[k].norm >= 0);
				bool texHere = (meshList[i].indexFace(j)[k].tex >= 0);
				if (vertHere) {
					stream << " " << (meshList[i].indexFace(j)[k].vert + vertsSoFar);
					if (normHere || texHere) stream << "/";
					if (texHere) stream << (meshList[i].indexFace(j)[k].tex + texsSoFar);
					if (normHere) stream << "/" << (meshList[i].indexFace(j)[k].norm + normsSoFar);
				}
			}
			stream << std::endl;
		}
		stream << "# " << meshList[i].faceCount() << " FACES" << std::endl << std::endl;
		vertsSoFar += meshList[i].vertextCount();
		normsSoFar += meshList[i].normalCount();
		texsSoFar += meshList[i].textureCount();
	}
	stream.close();
	return true;
}

