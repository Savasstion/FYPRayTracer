#include "Material.cuh"
#include <glm/detail/func_geometric.inl>


__host__ __device__ glm::vec3 Material::GetEmission() const
{
    return emissionColor * emissionPower;
}

__host__ __device__ float Material::GetEmissionRadiance() const
{
    return glm::length(emissionColor * emissionPower);
}

__host__ __device__ float Material::GetEmissionPower() const
{
    return emissionPower;
}
