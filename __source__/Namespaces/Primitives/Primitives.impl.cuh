#include"Primitives.cuh"
#include"../../DataStructures/Primitives/Pure/Vector3/Vector3.h"



namespace Primitives {



	__device__ __host__ inline PolyMesh sphere(int edges, float radius) {
		// Velidity assertion:
		if (edges < 4 || radius <= 0.0f) return PolyMesh();



		///////////////////////////////////////////
		// Setup:
		///////////////////////////////////////////

		// Constants for texture coordinate steps:
		const float texDeltaX = (1.0f / ((float)edges));
		const float texDeltaY = texDeltaX * 2.0f;

		// Constants for angular steps:
		const float delta = ((2.0f * PI) / ((float)edges));
		const int iMax = (edges - 1) / 2;
		
		// Sphere:
		PolyMesh mesh;



		///////////////////////////////////////////
		// Vertices:
		///////////////////////////////////////////

		// Upper Vertex:
		mesh.addVertex(Vertex(0, radius, 0));
		mesh.addNormal(Vector3::up());
		float tex = 0.0f;
		for (int i = 0; i < edges; i++) {
			mesh.addTexture(Vector3(tex, 1.0f, 0.0f));
			tex += texDeltaX;
		}
		
		// Middle Vertices:
		float alpha = PI * 0.5f;
		float texY = 1.0f;
		for (int i = 0; i < iMax; i++) {
			alpha -= delta;
			float sinAlpha = sin(alpha);
			float cosAlpha = cos(alpha);
			float beta = 0.0f;
			texY -= texDeltaY;
			tex = 0.0f;
			for (int j = 0; j < edges; j++) {
				Vector3 normal(cosAlpha * sin(beta), sinAlpha, cosAlpha * cos(beta));
				mesh.addVertex(normal * radius);
				mesh.addNormal(normal);
				mesh.addTexture(Vector3(tex, texY, 0.0f));
				beta += delta;
				tex += texDeltaX;
			}
			mesh.addTexture(Vector3(tex, texY, 0.0f));
		}

		// Lower Vertex:
		mesh.addVertex(Vertex(0, -radius, 0));
		mesh.addNormal(Vector3::down());
		tex = 0.0f;
		for (int i = 0; i < edges; i++) {
			mesh.addTexture(Vector3(tex, 0.0f, 0.0f));
			tex += texDeltaX;
		}

		

		///////////////////////////////////////////
		// Geometry:
		///////////////////////////////////////////

		// Top cap:
		for (int i = 0; i < edges; i++) {
			PolyMesh::IndexFace face;
			face.flush(3);
			face[0] = PolyMesh::IndexNode(1 + (i + 1) % edges, 1 + (i + 1) % edges, edges + i + 1);
			face[1] = PolyMesh::IndexNode(i + 1, i + 1, edges + i);
			face[2] = PolyMesh::IndexNode(0, 0, i);
			mesh.addFace(face);
		}

		// MiddleStrips:
		int vertUp, texUp, vertDown, texDown;
		for (int i = 0; i < (iMax - 1); i++) {
			vertUp = i * edges + 1;
			texUp = (i + 1) * (edges + 1) - 1;
			vertDown = vertUp + edges;
			texDown = texUp + edges + 1;
			for (int j = 0; j < edges; j++) {
				PolyMesh::IndexFace face;
				face.flush(4);
				face[0] = PolyMesh::IndexNode(vertUp + (j + 1) % edges, vertUp + (j + 1) % edges, texUp + j + 1);
				face[1] = PolyMesh::IndexNode(vertDown + (j + 1) % edges, vertDown + (j + 1) % edges, texDown + j + 1);
				face[2] = PolyMesh::IndexNode(vertDown + j, vertDown + j, texDown + j);
				face[3] = PolyMesh::IndexNode(vertUp + j, vertUp + j, texUp + j);
				mesh.addFace(face);
			}
		}

		// Lower cap:
		int lastVert = (mesh.vertextCount() - 1);
		int lowestTex = texDown + edges + 1;
		for (int i = 0; i < edges; i++) {
			PolyMesh::IndexFace face;
			face.flush(3);
			face[0] = PolyMesh::IndexNode(vertDown + (i + 1) % edges, vertDown + (i + 1) % edges, texDown + i + 1);
			face[1] = PolyMesh::IndexNode(lastVert, lastVert, lowestTex + i);
			face[2] = PolyMesh::IndexNode(vertDown + i, vertDown + i, texDown + i);
			mesh.addFace(face);
		}



		///////////////////////////////////////////
		// Return:
		///////////////////////////////////////////
		return mesh;
	}

}

