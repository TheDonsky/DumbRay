#pragma once

#include <QWidget>
#include "ui_RenderViewport.h"
#include "ImageWindow.h"
#include <thread>
#include <mutex>
#include <queue>
#include "../__source__/DataStructures/DumbRenderContext/DumbRenderContext.cuh"


class RenderViewport : public QWidget
{
	Q_OBJECT

public:
	RenderViewport(QWidget *parent = Q_NULLPTR, const char *filename = NULL);
	~RenderViewport();


protected:
	virtual void resizeEvent(QResizeEvent *event);


private:
	Ui::RenderViewport ui;
	ImageWindow window;
	DumbRenderContext context;
	DumbRenderContext::RenderInstance instance;
	std::mutex pixmapLock;
	QPixmap pixmap;

	std::string sceneFile;

	void emitUpdateImage(const QPixmap &map);
	static void updateImagePixmap(void *viewport, const QPixmap &map);

	enum State {
		STATE_DESTRUCTOR_CALLED = 1
	};
	volatile uint32_t flags;

	void loadScene();
	void continueRender();
	void interruptRender();
	void resetRender();

	enum Command {
		COMMAND_LOAD_SCENE,
		COMMAND_QUIT,
		COMMAND_START_RENDER,
		COMMAND_STOP_RENDER,
		COMMAND_RESTART_RENDER,
		COMMAND_UPDATE_TIMES,
		COMMAND_UPDATE_RESOURCES
	};
	std::mutex commandLock;
	std::condition_variable commandCondition;
	std::queue<Command> commandQueue;

	void dispatchCommand(Command command);
	template<Command command>
	inline void dispatchCommand() { dispatchCommand(command); }
	static void controlThread(RenderViewport *viewport);
	std::thread commandThread;
	static void maintenanceThread(RenderViewport *viewport);
	std::thread updateThread;


	enum ResolutionSettings {
		RESOLUTION_MATCH_VIEWPORT,
		RESOLUTION_FIXED_RELATIVE_SCALE,
		RESOLUTION_FIXED_CONSTANT_SCALE
	};
	float imageScale;
	ResolutionSettings resolutionSettings;

	void onIterationEnded();
	static void iterationEndedCallback(void *viewport);
	
	void updateResources();


signals:
	void updateImage();
	void issueSizeFix();
	void issueEnableButtons(bool enableStart, bool enableEnd, bool enableRestart);
	void issueRefreshIterationInfo(int iteration, double elapsed);
	void issueSetStatusText(const QString &text);

public: 
	signals:
		   void sourceFileChanged(const std::string &name);

private slots:
	void imageUpdated();
	void fixImageSize();
	void zoomPressed();
	void fitPressed();
	void zoomChanged(int sliderValue);
	void saveImage();
	void requestLoadScene();
	void setWidth(const QString &width);
	void setHeight(const QString &height);
	void enableButtons(bool enableStart, bool enableEnd, bool enableRestart);
	void refreshIterationInfo(int iteration, double elapsed);
	void startRender();
	void stopRender();
	void restartRender();
	void setStatusText(const QString &text);
	void cpuThreadCountChanged(int count);
};
