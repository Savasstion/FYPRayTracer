#include "ReSTIR_GI_Reservoir.cuh"
#include "../Utility/MathUtils.cuh"

__host__ __device__ bool ReSTIR_GI_Reservoir::UpdateReservoir(const PathSample& newSample, float newWeight, uint32_t& randSeed)
{
    weightFinal += newWeight;
    pathProcessedCount += 1;

    if (MathUtils::randomFloat(randSeed) < newWeight / weightFinal)
    {
        sample = newSample;
        return true;
    }
    return false;
}

__host__ __device__ bool ReSTIR_GI_Reservoir::MergeReservoir(const ReSTIR_GI_Reservoir& otherReservoir, float pdf, uint32_t& randSeed)
{
    uint32_t prevTotalCount = pathProcessedCount;
    bool sampleUpdated = UpdateReservoir(otherReservoir.sample, pdf * otherReservoir.weightFinal * otherReservoir.pathProcessedCount, randSeed);
    pathProcessedCount = prevTotalCount + otherReservoir.pathProcessedCount;

    return sampleUpdated;
}

__host__ __device__ void ReSTIR_GI_Reservoir::ResetReservoir()
{
    sample.ResetSample();

    weightSample = 0.0f;
    pathProcessedCount = 0;
    weightFinal = 0.0f;
}

__host__ __device__ void ReSTIR_GI_Reservoir::PathSample::ResetSample()
{
    visiblePoint = {0.0f, 0.0f, 0.0f};
    visibleNormal = {0.0f, 0.0f};  
    samplePoint = {0.0f, 0.0f, 0.0f};
    sampleNormal = {0.0f, 0.0f};  
    outgoingRadiance = {0.0f, 0.0f, 0.0f};   
    randSeed = 0;
}

__host__ __device__ bool ReSTIR_GI_Reservoir::CheckIfValid()
{
    return pathProcessedCount > 0;
}
