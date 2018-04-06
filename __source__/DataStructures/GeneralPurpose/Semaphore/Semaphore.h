#pragma once
#include <mutex>
#include <condition_variable>

class Semaphore {
public:
	Semaphore(unsigned int count = 0);
	void wait(int count = 1);
	void post(int count = 1);
	void set(unsigned int count);




private:
	unsigned volatile int value;
	std::mutex lock;
	std::condition_variable condition;
};

