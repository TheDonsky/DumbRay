#include "Semaphore.h"





Semaphore::Semaphore(unsigned int count) {
	value = count;
}
void Semaphore::wait(unsigned int count) {
	std::unique_lock<std::mutex> mutexLock(lock);
	while (value < count) condition.wait(mutexLock);
	value -= count;
}
void Semaphore::post(unsigned int count) {
	std::lock_guard<std::mutex> guard(lock);
	value += count;
	condition.notify_all();
}
void Semaphore::set(unsigned int count) {
	std::lock_guard<std::mutex> guard(lock);
	value = count;
	condition.notify_all();
}
