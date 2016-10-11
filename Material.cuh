#pragma once

#include"Photon.cuh"





template<typename HitType>
class Material{
public:
	struct HitInfo {
		Photon photon;
		Vector3 hitPoint;
		float distance;
		Vector3 cameraPosition;
	};





	struct ShaderReport {
		Photon reflectPhoton;
		Photon cameraPhoton;
	};





public:
	__host__ inline void init();
	template<typename Shader, typename... Args>
	__host__ inline bool init(const Args&... args);
	template<typename Shader>
	__host__ inline bool init(Shader *shader);
	__host__ inline bool dispose();

	
	
	
	__dumb__ ShaderReport cast(const HitType &object, const HitInfo &info)const;





public:
	typedef ShaderReport(*CastFunction)(void *shader, const HitType &object, const HitInfo &info);
	template<typename Shader>
	__dumb__ static ShaderReport castOnShader(void *shader, const HitType &object, const HitInfo &info);





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
