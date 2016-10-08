#pragma once

#include"Photon.cuh"





class Material{
public:
	struct HitInfo {
		Photon photon;
		Vector3 hitPoint;
		float distance;
		Vector3 cameraPosition;
	};
	struct HitInput {
		void *object;
		HitInfo input;
	};





	template<typename HitType>
	struct ShaderInput {
		HitType object;
		HitInfo info;
	};
	struct ShaderReport {
		Photon reflectPhoton;
		Photon cameraPhoton;
	};





public:
	__host__ inline void init();
	template<typename Shader, typename HitType, typename... Args>
	__host__ inline bool init(const Args&... args);
	template<typename Shader, typename HitType>
	__host__ inline bool init(Shader *shader);
	__host__ inline bool dispose();

	
	
	
	__dumb__ ShaderReport cast(const HitInput &hit)const;





public:
	typedef ShaderReport(*CastFunction)(void *shader, const HitInput &hit);
	template<typename Shader, typename HitType>
	__dumb__ static ShaderReport cast(void *shader, const HitInput &hit);





private:
	void *hostShader;
	void *devShader;
	bool ownsOnHost;
	
	CastFunction hostCast;
	CastFunction devCast;
	
	bool(*disposeOnHost)(void*&);
	bool(*disposeOnDevice)(void*&);





	template<typename Shader>
	__host__ static bool disposeFnHost(void *&shader);
	template<typename Shader>
	__host__ static bool disposeFnDevice(void *&shader);
};




#include"Material.impl.cuh"
