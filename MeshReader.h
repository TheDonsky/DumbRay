#pragma once
#include"PolyMesh.h"
#include"Pair.cuh"
#include"IntMap.h"
#include<fstream>
#include<map>


namespace MeshReader{
	inline static bool readObj(Stacktor<PolyMesh> &meshList, Stacktor<String> &nameList, std::string filename, bool dump =  false);
}


#include"MeshReader.impl.h"
