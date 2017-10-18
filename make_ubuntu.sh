nvcc -o DumbRay __source__/Test.cu \
	__source__/DataStructures/GeneralPurpose/Stacktor/Stacktor.test.cu \
	__source__/DataStructures/GeneralPurpose/IntMap/IntMap.test.cu \
	__source__/DataStructures/GeneralPurpose/Cutex/Cutex.test.cu \
	__source__/DataStructures/Objects/Scene/Raycasters/Octree/Octree.test.cu \
	__source__/DataStructures/GeneralPurpose/Generic/Generic.test.cu \
	__source__/DataStructures/GeneralPurpose/Handler/Handler.test.cu \
	__source__/DataStructures/GeneralPurpose/TypeTools/TypeTools.test.cu \
	__source__/DataStructures/Objects/Scene/SceneHandler/SceneHandler.test.cu \
	__source__/DataStructures/Renderers/Renderer/Renderer.test.cu

