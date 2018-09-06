#pragma once
#include "../../Namespaces/Dson/Dson.h"
#include "DumbRenderContextConnector.cuh"
#include "../Screen/Window/Window.cuh"


class DumbRenderContext {
public:
	DumbRenderContext();
	~DumbRenderContext();

	bool buildFromFile(const std::string &filename, std::ostream *errorStream);
	bool buildFromDson(const Dson::Object *object, std::ostream *errorStream);

	bool fromFile(const std::string &filename, std::ostream *errorStream);
	bool fromDson(const Dson::Object *object, std::ostream *errorStream);

	bool getImageId(const Dson::Object &object, int *imageId, std::ostream *errorStream);


	void runWindowRender();

	static void test();
	static void testFile(const std::string &filename);

	class RenderInstance {
	public:
		typedef void(*Callback)(void *aux);

		RenderInstance(DumbRenderContext *context, Window *window);
		~RenderInstance();

		void interruptRender();
		void uninterruptRender();
		bool renderInterrupted()const;

		void reset();

		void start();
		void stop();

		void setResolution(int width, int height);
		void getResolution(int &width, int &height);
		void getPixelColor(int x, int y, float &r, float &g, float &b, float &a)const;

		void onIterationComplete(Callback callback, void *aux);

		int iteration()const;
		double renderTime()const;

		int cpuThreads()const;
		void setCpuThreads(int count);

		int gpuCount()const;
		bool gpuOn(int index)const;
		void setGpu(int index, bool on);


	private:
		inline RenderInstance(const RenderInstance &) {}
		inline RenderInstance& operator=(const RenderInstance &) { return (*this); }

		DumbRenderContext *ctx;
		void *data;

		void initBuffer();
		void initRenderer();
		void initWindow();
		void initRenderContext();

		void iterationComplete();
		static void iterationComplete(void *reference);
	};
private:
	inline DumbRenderContext(const DumbRenderContext &) {}
	inline DumbRenderContext& operator=(const DumbRenderContext &) { return (*this); }

	void *data;
};
