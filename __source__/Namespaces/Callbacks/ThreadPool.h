#pragma once
#include <condition_variable>
#include <algorithm>
#include <thread>
#include <mutex>
#include <queue>

namespace Donsky {
    template<typename CallbackType>
    class ThreadPool {
    private:
        std::queue<CallbackType> callbacks;
        size_t threadCount;
        std::thread *workers;
        std::condition_variable condition;
        std::mutex lock;
        bool dontKillTillDone;
        volatile bool poolAlive;

        inline void worker() {
            while (true) {
                CallbackType call;
                {
                    std::unique_lock<std::mutex> guard(lock);
                    while (poolAlive && callbacks.empty()) condition.wait(guard);
                    if ((!poolAlive) && (!dontKillTillDone)) break;
                    if (callbacks.empty()) break;
                    call = callbacks.front();
                    callbacks.pop();
                }
                call();
            }
        }
        inline static void workerThread(ThreadPool *pool) { pool->worker(); }

    public:
        enum ThreadCount {
            THREADS_NONE = 0,
            THREADS_ALL = -1,
            THREADS_HALF = -2
        };

        inline ThreadPool(int numThreads = THREADS_ALL, bool killOnlyWhenDone = true) {
            threadCount = (size_t)
                        ((numThreads == THREADS_ALL) ? ((int)std::thread::hardware_concurrency()) : 
                        ((numThreads == THREADS_HALF) ? std::max((int)(std::thread::hardware_concurrency() / 2), (int)1) : 
                        std::max(numThreads, (int)0)));
            poolAlive = true;
            dontKillTillDone = killOnlyWhenDone;
            if (threadCount > 0) workers = new std::thread[threadCount];
            else workers = NULL;
            if (workers != NULL) {
                for (int i = 0; i < threadCount; i++) workers[i] = std::thread(workerThread, this);
            } 
            else threadCount = 0;
        }
        inline ~ThreadPool() {
            {
                std::lock_guard<std::mutex> guard(lock);
                poolAlive = false;
                condition.notify_all();
            }
            if (workers != NULL) {
                for (int i = 0; i < threadCount; i++) workers[i].join();
                delete[] workers;
            }
        }

        template<typename... ArgTypes>
        inline void schedule(ArgTypes... args) { schedule(CallbackType(args...)); }
        inline void schedule(const CallbackType &callback) { std::lock_guard<std::mutex> guard(lock); callbacks.push(callback); condition.notify_one(); }
        inline void schedule(CallbackType &&callback) { std::lock_guard<std::mutex> guard(lock); callbacks.push((CallbackType&&)callback); condition.notify_one(); }
    };
}
