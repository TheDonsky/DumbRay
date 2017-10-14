#pragma once
#include"../../DataStructures/Objects/Meshes/PolyMesh/PolyMesh.h"
#include"../../DataStructures/Primitives/Compound/Pair/Pair.cuh"
#include"../../DataStructures/GeneralPurpose/IntMap/IntMap.h"
#include<fstream>
#include<map>


namespace MeshReader{
	inline static bool readObj(Stacktor<PolyMesh> &meshList, Stacktor<String> &nameList, std::string filename, bool dump =  false);
	inline static bool writeObj(const Stacktor<PolyMesh> &meshList, const Stacktor<String> &nameList, std::string filename);
}


#include"MeshReader.impl.h"
