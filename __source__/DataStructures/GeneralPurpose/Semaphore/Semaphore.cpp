#include "Semaphore.h"





Semaphore::Semaphore(unsigned int count) {
	value = count;
}
void Semaphore::wait() {
	std::unique_lock<std::mutex> mutexLock(lock);
	while (value <= 0) condition.wait(mutexLock);
	value--;
}
void Semaphore::post() {
	std::lock_guard<std::mutex> guard(lock);
	value++;
	condition.notify_one();
}
void Semaphore::set(unsigned int count) {
	std::lock_guard<std::mutex> guard(lock);
	value = count;
	condition.notify_all();
}
