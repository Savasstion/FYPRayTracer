#include "ReSTIR_DI_Reservoir.cuh"

__host__ __device__ bool ReSTIR_DI_Reservoir::UpdateReservoir(uint32_t candidateEmissiveIndex, float weight,
                                                              uint32_t& randSeed)
{
    //  the parameter 'weight' passed in should be the amount of light contribution calculated using the rendering equation without the Visibility Term

    weightSum += weight;
    emissiveProcessedCount += 1;

    if (MathUtils::randomFloat(randSeed) < weight / weightSum)
    {
        indexEmissive = candidateEmissiveIndex;
        return true;
    }

    return false;
}

__host__ __device__ bool ReSTIR_DI_Reservoir::UpdateReservoir(uint32_t candidateEmissiveIndex, float weight, uint32_t count, uint32_t& randSeed)
{
    //  the parameter 'weight' passed in should be the amount of light contribution calculated using the rendering equation without the Visibility Term

    weightSum += weight;
    emissiveProcessedCount += count;

    if (MathUtils::randomFloat(randSeed) < weight / weightSum)
    {
        indexEmissive = candidateEmissiveIndex;
        return true;
    }

    return false;
}

__host__ __device__ void ReSTIR_DI_Reservoir::ResetReservoir()
{
    indexEmissive = 0;
    weightEmissive = 0.0f;
    weightSum = 0.0f;
    emissiveProcessedCount = 0;
}
