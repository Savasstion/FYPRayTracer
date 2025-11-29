#include "ReSTIR_GI_Reservoir.cuh"
#include "../Utility/MathUtils.cuh"

__host__ __device__ bool ReSTIR_GI_Reservoir::UpdateReservoir(PathSample newSample, float newWeight, uint32_t& randSeed)
{
    weightSum += newWeight;
    pathProcessedCount += 1;

    if (MathUtils::randomFloat(randSeed) < newWeight / weightSum)
    {
        sample = newSample;
        return true;
    }
    return false;
}

__host__ __device__ bool ReSTIR_GI_Reservoir::MergeReservoir(ReSTIR_GI_Reservoir otherReservoir, float pdf, uint32_t& randSeed)
{
    uint32_t prevTotalCount = pathProcessedCount;
    bool sampleUpdated = UpdateReservoir(otherReservoir.sample, pdf * otherReservoir.weightSum * otherReservoir.pathProcessedCount, randSeed);
    pathProcessedCount = prevTotalCount + otherReservoir.pathProcessedCount;

    return sampleUpdated;
}

__host__ __device__ void ReSTIR_GI_Reservoir::ResetReservoir()
{
    sample.visiblePoint = {0.0f, 0.0f, 0.0f};
    sample.visibleNormal = {0.0f, 0.0f};  
    sample.samplePoint = {0.0f, 0.0f, 0.0f};
    sample.sampleNormal = {0.0f, 0.0f};  
    sample.outgoingRadiance = {0.0f, 0.0f, 0.0f};   
    sample.randSeed = 0;

    weightSample = 0.0f;
    pathProcessedCount = 0;
    weightSum = 0.0f;
}

__host__ __device__ bool ReSTIR_GI_Reservoir::CheckIfValid()
{
    return pathProcessedCount > 0;
}
