#pragma once
#include <stddef.h>

namespace Donsky {
	namespace Function {
#ifndef __device__
#define __device__
#define __DEVICE_WAS_NOT_DEFINED__
#endif

#ifndef __host__
#define __host__
#define __HOST_WAS_NOT_DEFINED__
#endif

		template <typename... Types>
		class Arguments {};

		template <typename Type>
		class Arguments<Type> {
		public:
			__device__ __host__ inline Arguments() {}
			__device__ __host__ inline Arguments(Type val) : value(val) {}
			Type value;
		};

		template <typename FirstType, typename SecondType, typename... RestTypes>
		class Arguments<FirstType, SecondType, RestTypes...> {
		public:
			__device__ __host__ inline Arguments() {}
			__device__ __host__ inline Arguments(FirstType val) : value(val) {}
			__device__ __host__ inline Arguments(FirstType val, SecondType nextVal, RestTypes... vals) : value(val), next(nextVal, vals...) {}
			FirstType value;
			Arguments<SecondType, RestTypes...> next;
		};

		template <typename ReturnType, typename... Types>
		__device__ __host__ inline ReturnType call(ReturnType (*function)(Types...), Types... args) {
			return function(args...);
		}
		template <typename ReturnType, typename... ArgTypes, typename... ExpandedArgTypes>
		__device__ __host__ inline ReturnType call(ReturnType (*function)(ArgTypes...), const Arguments<> &args, ExpandedArgTypes... expanded) {
			return call(function, expanded...);
		}
		template <typename ReturnType, typename... ArgTypes, typename FirstArg, typename... ExpandedArgTypes>
		__device__ __host__ inline ReturnType call(ReturnType (*function)(ArgTypes...), const Arguments<FirstArg> &args, ExpandedArgTypes... expanded) {
			return call(function, expanded..., args.value);
		}
		template <typename ReturnType, typename... ArgTypes, typename FirstArg, typename... ArgumentsTypes, typename... ExpandedArgTypes>
		__device__ __host__ inline ReturnType call(ReturnType (*function)(ArgTypes...), const Arguments<FirstArg, ArgumentsTypes...> &args, ExpandedArgTypes... expanded) {
			return call(function, args.next, expanded..., args.value);
		}

		template <typename Class, typename ReturnType, typename... Types>
		__device__ __host__ inline ReturnType call(Class *obj, ReturnType (Class::*function)(Types...), Types... args) {
			return (obj->*function)(args...);
		}
		template <typename Class, typename ReturnType, typename... Types>
		__device__ __host__ inline ReturnType call(Class &obj, ReturnType (Class::*function)(Types...), Types... args) {
			return (obj.*function)(args...);
		}
		template <typename Class, typename ReturnType, typename... ArgTypes, typename FirstArg, typename... ExpandedArgTypes>
		__device__ __host__ inline ReturnType call(Class *obj, ReturnType (Class::*function)(ArgTypes...), const Arguments<FirstArg> &args, ExpandedArgTypes... expanded) {
			return call(obj, function, expanded..., args.value);
		}
		template <typename Class, typename ReturnType, typename... ArgTypes, typename FirstArg, typename... ArgumentsTypes, typename... ExpandedArgTypes>
		__device__ __host__ inline ReturnType call(Class *obj, ReturnType (Class::*function)(ArgTypes...), const Arguments<FirstArg, ArgumentsTypes...> &args, ExpandedArgTypes... expanded) {
			return call(obj, function, args.next, expanded..., args.value);
		}
		template <typename Class, typename ReturnType>
		__device__ __host__ inline ReturnType call(Class *obj, ReturnType (Class::*function)(), const Arguments<> &args) {
			return (obj->*function)();
		}
		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline ReturnType call(Class &obj, ReturnType (Class::*function)(ArgTypes...), const Arguments<ArgTypes...> &args) {
			return call(&obj, function, args);
		}

		template <typename Class, typename ReturnType, typename... Types>
		__device__ __host__ inline ReturnType call(const Class *obj, ReturnType (Class::*function)(Types...) const, Types... args) {
			return (obj->*function)(args...);
		}
		template <typename Class, typename ReturnType, typename... Types>
		__device__ __host__ inline ReturnType call(const Class &obj, ReturnType (Class::*function)(Types...) const, Types... args) {
			return (obj.*function)(args...);
		}
		template <typename Class, typename ReturnType, typename... ArgTypes, typename FirstArg, typename... ExpandedArgTypes>
		__device__ __host__ inline ReturnType call(const Class *obj, ReturnType (Class::*function)(ArgTypes...) const, const Arguments<FirstArg> &args, ExpandedArgTypes... expanded) {
			return call(obj, function, expanded..., args.value);
		}
		template <typename Class, typename ReturnType, typename... ArgTypes, typename FirstArg, typename... ArgumentsTypes, typename... ExpandedArgTypes>
		__device__ __host__ inline ReturnType call(const Class *obj, ReturnType (Class::*function)(ArgTypes...) const, const Arguments<FirstArg, ArgumentsTypes...> &args, ExpandedArgTypes... expanded) {
			return call(obj, function, args.next, expanded..., args.value);
		}
		template <typename Class, typename ReturnType>
		__device__ __host__ inline ReturnType call(const Class *obj, ReturnType (Class::*function)() const, const Arguments<> &args) {
			return (obj->*function)();
		}
		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline ReturnType call(const Class &obj, ReturnType (Class::*function)(ArgTypes...) const, const Arguments<ArgTypes...> &args) {
			return call(&obj, function, args);
		}

		template <typename ReturnType>
		class Callable {
		public:
			virtual ~Callable() {}
			virtual ReturnType operator()() = 0;
		};

		template <typename Class, typename ReturnType, typename... ArgTypes>
		class Call : public Callable<ReturnType> {
		private:
			Class *instance;
			ReturnType (Class::*function)(ArgTypes...);
			Arguments<ArgTypes...> arguments;

		public:
			__device__ __host__ inline Call() {}
			__device__ __host__ inline Call(const Call &other) : instance(other.instance), function(other.function), arguments(other.arguments) {}
			__device__ __host__ inline Call& operator=(const Call &other) { instance = other.instance; function = other.function; arguments = other.arguments; return (*this); }
			__device__ __host__ inline Call(Class *obj, ReturnType (Class::*fn)(ArgTypes...), ArgTypes... args) : Call(obj, fn, Arguments<ArgTypes...>(args...)) {}
			__device__ __host__ inline Call(Class *obj, ReturnType (Class::*fn)(ArgTypes...), const Arguments<ArgTypes...> &args) : instance(obj), function(fn), arguments(args) {}
			__device__ __host__ inline ReturnType operator()() { return call(instance, function, arguments); }
		};

		template <typename Class, typename ReturnType, typename... ArgTypes>
		class Call<const Class, ReturnType, ArgTypes...> : public Callable<ReturnType> {
		private:
			const Class *instance;
			ReturnType (Class::*function)(ArgTypes...) const;
			Arguments<ArgTypes...> arguments;

		public:
			__device__ __host__ inline Call() {}
			__device__ __host__ inline Call(const Call &other) : instance(other.instance), function(other.function), arguments(other.arguments) {}
			__device__ __host__ inline Call& operator=(const Call &other) { instance = other.instance; function = other.function; arguments = other.arguments; return (*this); }
			__device__ __host__ inline Call(const Class *obj, ReturnType (Class::*fn)(ArgTypes...) const, ArgTypes... args) : Call(obj, fn, Arguments<ArgTypes...>(args...)) {}
			__device__ __host__ inline Call(const Class *obj, ReturnType (Class::*fn)(ArgTypes...) const, const Arguments<ArgTypes...> &args) : instance(obj), function(fn), arguments(args) {}
			__device__ __host__ inline ReturnType operator()() { return call(instance, function, arguments); }
		};

		template <typename ReturnType, typename... ArgTypes>
		class Call<void, ReturnType, ArgTypes...> : public Callable<ReturnType> {
		private:
			ReturnType (*function)(ArgTypes...);
			Arguments<ArgTypes...> arguments;

		public:
			__device__ __host__ inline Call() {}
			__device__ __host__ inline Call(const Call &other) : function(other.function), arguments(other.arguments) {}
			__device__ __host__ inline Call& operator=(const Call &other) { function = other.function; arguments = other.arguments; return (*this); }
			__device__ __host__ inline Call(ReturnType (*fn)(ArgTypes...), ArgTypes... args) : Call(fn, Arguments<ArgTypes...>(args...)) {}
			__device__ __host__ inline Call(ReturnType (*fn)(ArgTypes...), const Arguments<ArgTypes...> &args) : function(fn), arguments(args) {}
			__device__ __host__ inline ReturnType operator()() { return call(function, arguments); }
		};

		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<Class, ReturnType, ArgTypes...> callback(Class *object, ReturnType (Class::*function)(ArgTypes...), const Arguments<ArgTypes...> &arguments) {
			return Call<Class, ReturnType, ArgTypes...>(object, function, arguments);
		}
		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<Class, ReturnType, ArgTypes...> callback(Class *object, ReturnType (Class::*function)(ArgTypes...), ArgTypes... arguments) {
			return Call<Class, ReturnType, ArgTypes...>(object, function, Arguments<ArgTypes...>(arguments...));
		}
		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<Class, ReturnType, ArgTypes...> callback(Class &object, ReturnType (Class::*function)(ArgTypes...), const Arguments<ArgTypes...> &arguments) {
			return Call<Class, ReturnType, ArgTypes...>(&object, function, arguments);
		}
		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<Class, ReturnType, ArgTypes...> callback(Class &object, ReturnType (Class::*function)(ArgTypes...), ArgTypes... arguments) {
			return Call<Class, ReturnType, ArgTypes...>(&object, function, Arguments<ArgTypes...>(arguments...));
		}

		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<const Class, ReturnType, ArgTypes...> callback(const Class *object, ReturnType (Class::*function)(ArgTypes...) const, const Arguments<ArgTypes...> &arguments) {
			return Call<const Class, ReturnType, ArgTypes...>(object, function, arguments);
		}
		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<const Class, ReturnType, ArgTypes...> callback(const Class *object, ReturnType (Class::*function)(ArgTypes...) const, ArgTypes... arguments) {
			return Call<const Class, ReturnType, ArgTypes...>(object, function, Arguments<ArgTypes...>(arguments...));
		}
		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<const Class, ReturnType, ArgTypes...> callback(const Class &object, ReturnType (Class::*function)(ArgTypes...) const, const Arguments<ArgTypes...> &arguments) {
			return Call<const Class, ReturnType, ArgTypes...>(&object, function, arguments);
		}
		template <typename Class, typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<const Class, ReturnType, ArgTypes...> callback(const Class &object, ReturnType (Class::*function)(ArgTypes...) const, ArgTypes... arguments) {
			return Call<const Class, ReturnType, ArgTypes...>(&object, function, Arguments<ArgTypes...>(arguments...));
		}

		template <typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<void, ReturnType, ArgTypes...> callback(void *, ReturnType (*function)(ArgTypes...), const Arguments<ArgTypes...> &arguments) {
			return Call<void, ReturnType, ArgTypes...>(function, arguments);
		}
		template <typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<void, ReturnType, ArgTypes...> callback(void *, ReturnType (*function)(ArgTypes...), ArgTypes... arguments) {
			return Call<void, ReturnType, ArgTypes...>(function, Arguments<ArgTypes...>(arguments...));
		}
		template <typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<void, ReturnType, ArgTypes...> callback(ReturnType (*function)(ArgTypes...), const Arguments<ArgTypes...> &arguments) {
			return Call<void, ReturnType, ArgTypes...>(function, arguments);
		}
		template <typename ReturnType, typename... ArgTypes>
		__device__ __host__ inline Call<void, ReturnType, ArgTypes...> callback(ReturnType (*function)(ArgTypes...), ArgTypes... arguments) {
			return Call<void, ReturnType, ArgTypes...>(function, Arguments<ArgTypes...>(arguments...));
		}

		template <typename ReturnType>
		class Callback : public Callable<ReturnType> {
		private:
			Callable<ReturnType> *function;
			Callable<ReturnType> *(*cloneFn)(const Callable<ReturnType> *);

			template <typename ClassType>
			__device__ __host__ inline static Callable<ReturnType> *clone(const Callable<ReturnType> *function) { return new ClassType(*((const ClassType *)function)); }

			__device__ __host__ inline void clear() { if (function != NULL) { delete function; function = NULL; } }

		public:
			__device__ __host__ inline Callback() : function(NULL) {}

			template <typename CallableType>
			__device__ __host__ inline Callback(const CallableType &callable) : function(clone<CallableType>(&callable)), cloneFn(clone<CallableType>) { }

			template <typename... ArgTypes>
			__device__ __host__ inline Callback(ReturnType (*function)(ArgTypes...), const Arguments<ArgTypes...> &args) : Callback(callback(function, args)) {}
			template <typename... ArgTypes>
			__device__ __host__ inline Callback(ReturnType (*function)(ArgTypes...), ArgTypes... args) : Callback(function, Arguments<ArgTypes...>(args...)) {}
			template <typename... ArgTypes>
			__device__ __host__ inline Callback(void *, ReturnType (*function)(ArgTypes...), const Arguments<ArgTypes...> &args) : Callback(callback(function, args)) {}
			template <typename... ArgTypes>
			__device__ __host__ inline Callback(void *, ReturnType (*function)(ArgTypes...), ArgTypes... args) : Callback(function, Arguments<ArgTypes...>(args...)) {}

			template <typename Class, typename... ArgTypes>
			__device__ __host__ inline Callback(Class *obj, ReturnType (Class::*function)(ArgTypes...), const Arguments<ArgTypes...> &args) : Callback(callback(obj, function, args)) {}
			template <typename Class, typename... ArgTypes>
			__device__ __host__ inline Callback(Class *obj, ReturnType (Class::*function)(ArgTypes...), ArgTypes... args) : Callback(obj, function, Arguments<ArgTypes...>(args...)) {}
			template <typename Class, typename... ArgTypes>
			__device__ __host__ inline Callback(Class &obj, ReturnType (Class::*function)(ArgTypes...), const Arguments<ArgTypes...> &args) : Callback(callback(obj, function, args)) {}
			template <typename Class, typename... ArgTypes>
			__device__ __host__ inline Callback(Class &obj, ReturnType (Class::*function)(ArgTypes...), ArgTypes... args) : Callback(obj, function, Arguments<ArgTypes...>(args...)) {}

			template <typename Class, typename... ArgTypes>
			__device__ __host__ inline Callback(const Class *obj, ReturnType (Class::*function)(ArgTypes...) const, const Arguments<ArgTypes...> &args) : Callback(callback(obj, function, args)) {}
			template <typename Class, typename... ArgTypes>
			__device__ __host__ inline Callback(const Class *obj, ReturnType (Class::*function)(ArgTypes...) const, ArgTypes... args) : Callback(obj, function, Arguments<ArgTypes...>(args...)) {}
			template <typename Class, typename... ArgTypes>
			__device__ __host__ inline Callback(const Class &obj, ReturnType (Class::*function)(ArgTypes...) const, const Arguments<ArgTypes...> &args) : Callback(callback(obj, function, args)) {}
			template <typename Class, typename... ArgTypes>
			__device__ __host__ inline Callback(const Class &obj, ReturnType (Class::*function)(ArgTypes...) const, ArgTypes... args) : Callback(obj, function, Arguments<ArgTypes...>(args...)) {}

			__device__ __host__ inline Callback(const Callback &other) : function(NULL) { (*this) = other; }
			__device__ __host__ inline Callback &operator=(const Callback &other) {
				if ((&other) != this) {
					clear();
					if (other.function != NULL) function = other.cloneFn(other.function);
					cloneFn = other.cloneFn;
				}
				return (*this);
			}

			__device__ __host__ inline Callback(Callback &&other) : function(other.function), cloneFn(other.cloneFn) { other.function = NULL; }
			__device__ __host__ inline Callback &operator=(Callback &&other) {
				if ((&other) != this) {
					clear();
					function = other.function;
					cloneFn = other.cloneFn;
					other.function = NULL;
				}
				return (*this);
			}

			__device__ __host__ inline ~Callback() { clear(); }

			__device__ __host__ inline bool callable() { return (function != NULL); }
			__device__ __host__ inline ReturnType operator()() { return function->operator()(); }
		};

#ifdef __DEVICE_WAS_NOT_DEFINED__
#undef __device__
#undef __DEVICE_WAS_NOT_DEFINED__
#endif

#ifdef __HOST_WAS_NOT_DEFINED__
#undef __host__
#undef __HOST_WAS_NOT_DEFINED__
#endif
	} // namespace Function
} // namespace Donsky