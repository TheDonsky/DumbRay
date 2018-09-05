#pragma once
#include<mutex>

class ReaderWriterLock {
private:
	std::mutex lock;
	std::condition_variable cond;
	volatile uint32_t readers;

public:
	inline ReaderWriterLock() : readers(0) {}

	inline void lockRead() { std::lock_guard<std::mutex> guard(lock); readers++; }
	inline void unlockRead() { std::lock_guard<std::mutex> guard(lock); readers--; cond.notify_all(); }

	inline void lockWrite() {
		std::unique_lock<std::mutex> guard(lock);
		while (readers > 0) cond.wait(guard);
		guard.release();
	}
	inline void unlockWrite() { lock.unlock(); }

	class ReadLock {
	private:
		ReaderWriterLock *lock;

	public:
		inline ReadLock(ReaderWriterLock *lck) : lock(lck) { lock->lockRead(); }
		inline ~ReadLock() { lock->unlockRead(); }
	};

	class WriteLock {
	private:
		ReaderWriterLock *lock;

	public:
		inline WriteLock(ReaderWriterLock *lck) : lock(lck) { lock->lockWrite(); }
		inline ~WriteLock() { lock->unlockWrite(); }
	};
};
