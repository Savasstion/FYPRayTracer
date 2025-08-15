#ifndef MATERIAL_CUH
#define MATERIAL_CUH

struct Material
{
    glm::vec3 albedo{1.0f};
    float roughness = 1.0f;
    float metallic = 0.0f;
    glm::vec3 emissionColor{0.0f};
    float emissionPower = 0.0f;

    __host__ __device__ glm::vec3 GetEmission() const {return emissionColor * emissionPower;}
    __host__ __device__ float GetEmissionPower() const {return emissionPower;}
};

#endif
