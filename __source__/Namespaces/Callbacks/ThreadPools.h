#pragma once
#include "ThreadPool.h"
#include "Function.h"

namespace Donsky {
    namespace ThreadPools {
        typedef Function::Callback<void> Callback;
        typedef Donsky::ThreadPool<Callback> ThreadPool;
        typedef Donsky::ThreadPool<Function::Call<void, void> > FunctionThreadPool;
        template<typename Class, typename... ArgTypes> class MethodThreadPool : public Donsky::ThreadPool<Function::Call<Class, void, ArgTypes...> > {};
    }
}