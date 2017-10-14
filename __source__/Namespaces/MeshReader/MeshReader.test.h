#pragma once
#include "MeshReader.h"
#include "../Primitives/Primitives.cuh"


namespace MeshReaderTest {
	// ########################################
	// ############ READING FILES: ############
	// ########################################
	inline static void readMeshes(Stacktor<PolyMesh> &meshes) {
		Stacktor<String> names;
		while (true) {
			std::string line;
			std::cout << "ENTER .obj FILE TO READ(EMPTY LINE TO MOVE ON): ";
			std::getline(std::cin, line);
			int start = meshes.size();
			if (line == "") break;
			else if (line == "Primitives::") {
				std::cout << "    Type and params: ";
				std::cin >> line;
				if (line == "sphere") {
					int n; std::cin >> n;
					float r; std::cin >> r;
					meshes.push(Primitives::sphere(n, r));
					names.push("Primitives::Sphere()");
					std::getline(std::cin, line);
				}
				else std::cout << "ERROR: Unknown type: " << line << std::endl;
			}
			else {
				std::cout << "READING FILE... PLEASE WAIT" << std::endl;
				if (!MeshReader::readObj(meshes, names, line)) {
					std::cout << "UNABLE TO READ THE FILE...." << std::endl;
					continue;
				}
				else {
					std::cout << "FILE (" << line << ") CONTENT: " << std::endl;
					for (int i = start; i < meshes.size(); i++)
						std::cout << "    NAME: " << (names[i] + 0) << "; V: " << meshes[i].vertextCount() << "; N: " << meshes[i].normalCount() << "; T: " << meshes[i].textureCount() << "; F: " << meshes[i].faceCount() << std::endl;
				}
			}
			while (true) {
				std::cout << "WOULD YOU LIKE TO TRANSFORM THE CONTENT(y/n)? ";
				std::string answer; std::getline(std::cin, answer);
				if (answer == "y" || answer == "Y" || answer == "yes" || answer == "Yes" || answer == "YES") {
					std::cout << "ENTER TRANSFORM POSITION: ";
					Vector3 pos; std::cin >> pos;
					std::cout << "ENTER EULER ANGLES: ";
					Vector3 euler; std::cin >> euler;
					std::cout << "ENTER SCALE: ";
					Vector3 scale; std::cin >> scale;
					std::string lineTillEnd;
					std::getline(std::cin, lineTillEnd);
					Transform trans(pos, euler, scale);
					Transform normalTrans(Vector3(0, 0, 0), euler, Vector3(1, 1, 1));
					for (int i = start; i < meshes.size(); i++) {
						for (int j = 0; j < meshes[i].vertextCount(); j++)
							meshes[i].vertex(j) >>= trans;
						for (int j = 0; j < meshes[i].normalCount(); j++)
							meshes[i].normal(j) >>= normalTrans;
					}
					break;
				}
				else if (answer == "" || answer == "n" || answer == "N" || answer == "no" || answer == "No" || answer == "NO") break;
				else std::cout << "YOU'RE NOT ANSWERING MY QUESTIONS, ARE YOU?" << std::endl;
			}
		}
		int totalVerts = 0, totalNorms = 0, totalTexts = 0, totalFaces = 0, totalTriangles = 0;
		for (int i = 0; i < meshes.size(); i++) {
			totalVerts += meshes[i].vertextCount();
			totalNorms += meshes[i].normalCount();
			totalTexts += meshes[i].textureCount();
			totalFaces += meshes[i].faceCount();
			for (int j = 0; j < meshes[i].faceCount(); j++)
				totalTriangles += (meshes[i].indexFace(j).size() - 2);
		}
		std::cout << std::endl << "SCENE TOTAL: " << std::endl;
		std::cout << "   VERTICES:              " << totalVerts << std::endl;
		std::cout << "   VERTEX NORMALS:        " << totalNorms << std::endl;
		std::cout << "   VERTEX TEXTURES:       " << totalTexts << std::endl;
		std::cout << "   FACES:                 " << totalFaces << std::endl;
		std::cout << "   TRIANGLES(from faces): " << totalTriangles << std::endl;
		std::cout << std::endl;
	}
}

