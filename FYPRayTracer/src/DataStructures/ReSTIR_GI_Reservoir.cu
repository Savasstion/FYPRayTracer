#include "ReSTIR_GI_Reservoir.cuh"
#include <glm/gtx/quaternion.hpp>
#include "../Utility/MathUtils.cuh"

__host__ __device__ bool ReSTIR_GI_Reservoir::UpdateReservoir(const PathSample& newSample, float newWeight, uint32_t& randSeed)
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

__host__ __device__ bool ReSTIR_GI_Reservoir::UpdateReservoir(const PathSample& newSample, float newWeight, uint32_t count,
    uint32_t& randSeed)
{
    weightSum += newWeight;
    pathProcessedCount += count;

    if (MathUtils::randomFloat(randSeed) < newWeight / weightSum)
    {
        sample = newSample;
        return true;
    }
    return false;
}

__host__ __device__ bool ReSTIR_GI_Reservoir::MergeReservoir(const ReSTIR_GI_Reservoir& otherReservoir, float pdf, uint32_t& randSeed)
{
    uint32_t prevTotalCount = pathProcessedCount;
    bool sampleUpdated = UpdateReservoir(otherReservoir.sample, pdf * otherReservoir.weightSum * otherReservoir.pathProcessedCount, randSeed);
    pathProcessedCount = prevTotalCount + otherReservoir.pathProcessedCount;

    return sampleUpdated;
}

__host__ __device__ void ReSTIR_GI_Reservoir::ResetReservoir()
{
    sample.ResetSample();

    weightSample = 0.0f;
    pathProcessedCount = 0;
    weightSum = 0.0f;
}

__host__ __device__ void ReSTIR_GI_Reservoir::PathSample::ResetSample()
{
    visiblePoint = {0.0f, 0.0f, 0.0f};
    visibleNormal = {0.0f, 0.0f};  
    samplePoint = {0.0f, 0.0f, 0.0f};
    sampleNormal = {0.0f, 0.0f};  
    outgoingRadiance = {0.0f, 0.0f, 0.0f};   
    randSeed = 0;
    samplePDF = 0.0f;
}

__host__ __device__ bool ReSTIR_GI_Reservoir::CheckIfValid() const
{
    return pathProcessedCount > 0 && glm::length2(sample.outgoingRadiance) > 0.0f;
}
