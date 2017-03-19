#pragma once
#include"PolyMesh.h"
#include"Pair.cuh"
#include"IntMap.h"
#include<fstream>
#include<map>


namespace MeshReader{
	inline static bool readObj(Stacktor<PolyMesh> &meshList, Stacktor<String> &nameList, std::string filename, bool dump =  false);
	inline static bool writeObj(const Stacktor<PolyMesh> &meshList, const Stacktor<String> &nameList, std::string filename);
}


#include"MeshReader.impl.h"
