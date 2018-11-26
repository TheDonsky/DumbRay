nvcc -rdc=true -dc -ccbin g++-6 -std=c++14 \
	../__source__/DataStructures/DumbRenderContext/DumbRenderContext.cu \
	../__source__/DataStructures/DumbRenderContext/DumbRenderContextConnector.cu \
	../__source__/DataStructures/DumbRenderContext/DumbRenderContextRegistry.cu \
	../__source__/DataStructures/GeneralPurpose/Cutex/Cutex.test.cu \
	../__source__/DataStructures/GeneralPurpose/DumbRand/DumbRand.cu \
	../__source__/DataStructures/GeneralPurpose/DumbRand/DumbRand.test.cu \
	../__source__/DataStructures/GeneralPurpose/Generic/Generic.test.cu \
	../__source__/DataStructures/GeneralPurpose/Handler/Handler.test.cu \
	../__source__/DataStructures/GeneralPurpose/IntMap/IntMap.test.cu \
	../__source__/DataStructures/GeneralPurpose/Semaphore/Semaphore.cu \
	../__source__/DataStructures/GeneralPurpose/Stacktor/Stacktor.test.cu \
	../__source__/DataStructures/GeneralPurpose/TypeTools/TypeTools.test.cu \
	../__source__/DataStructures/Objects/Components/DumbStructs.cu \
	../__source__/DataStructures/Objects/Scene/Raycasters/Octree/Octree.test.cu \
	../__source__/DataStructures/Renderers/BlockRenderer/BlockRenderer.cu \
	../__source__/DataStructures/Renderers/BufferedRenderer/BufferedRenderer.cu \
	../__source__/DataStructures/Renderers/BufferedRenderProcess/BufferedRenderProcess.cu \
	../__source__/DataStructures/Renderers/BufferedRenderProcess/BufferedRenderProcess.test.cu \
	../__source__/DataStructures/Renderers/DumbRenderer/DumbRenderer.cu \
	../__source__/DataStructures/Renderers/DumbRenderer/DumbRenderer.test.cu \
	../__source__/DataStructures/Renderers/Renderer/Renderer.cu \
	../__source__/DataStructures/Renderers/Renderer/Renderer.test.cu \
	../__source__/DataStructures/Screen/BufferedWindow/BufferedWindow.cu \
	../__source__/DataStructures/Screen/FrameBuffer/BlockBasedFrameBuffer/BlockBasedFrameBuffer.impl.cu \
	../__source__/DataStructures/Screen/FrameBuffer/BlockBasedFrameBuffer/BlockBasedFrameBuffer.test.cu \
	../__source__/DataStructures/Screen/FrameBuffer/MemoryMappedFrameBuffer/MemoryMappedFrameBuffer.test.cu \
	../__source__/DataStructures/Screen/FrameBuffer/FrameBuffer.test.cu \
	../__source__/DataStructures/Screen/Window/WindowsWindow.cu \
	../__source__/Namespaces/Device/Device.cu \
	../__source__/Namespaces/Dson/Dson.cu \
	../__source__/Namespaces/Dson/Dson.test.cu \
	../__source__/Namespaces/Images/Images.cu \
	../__source__/Namespaces/Images/Images.test.cu \
	../__source__/Namespaces/Images/lodepng.cu \
	../__source__/Namespaces/TextParser/ByteSet.cu \
	../__source__/Namespaces/TextParser/Error.cu \
	../__source__/Namespaces/TextParser/ParserNetwork.cu \
	../__source__/Namespaces/TextParser/RecursiveParser.cu \
	../__source__/Namespaces/Windows/Windows.cu \
	../__source__/Playground/Checkerboard/Checkerboard.cu \
	../__source__/TestingModule.cu \
	../__source__/Main.cu && nvcc *.o -o=__DumbRay__


	

