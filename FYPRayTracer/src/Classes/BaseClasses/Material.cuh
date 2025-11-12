#ifndef MATERIAL_CUH
#define MATERIAL_CUH
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <glm/vec3.hpp>

struct Material
{
    bool isUseAlbedoMap = false;
    glm::vec3 albedo{1.0f,0.0f,1.0f};
    uint32_t albedoMapIndex = static_cast<uint32_t>(-1);
    
    float roughness = 1.0f;
    float metallic = 0.0f;
    glm::vec3 emissionColor{0.0f};
    float emissionPower = 0.0f;

    __host__ __device__ glm::vec3 GetEmission() const;
    __host__ __device__ float GetEmissionRadiance() const;
    __host__ __device__ float GetEmissionPower() const;
};

#endif
