#ifndef RESTIR_DI_RESERVOIR_CUH
#define RESTIR_DI_RESERVOIR_CUH
#include <cstdint>
#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include "../Utility/MathUtils.cuh"

struct ReSTIR_DI_Reservoir
{
    uint32_t indexEmissive;
    // index of most important light, the index stores here refers to the index of emissiveTriangle list
    float weightEmissive; // light weight
    float emissivePDF = 0.0f;   //  pdf or radiance length of emissive light
    float weightSum; // sum of all weights for all lights processed
    uint32_t emissiveProcessedCount; // number of lights processed for this reservoir

    __host__ __device__ bool UpdateReservoir(uint32_t candidateEmissiveIndex, float weight, float pdf, uint32_t& randSeed);
    __host__ __device__ bool UpdateReservoir(uint32_t candidateEmissiveIndex, float weight, uint32_t count, float pdf, uint32_t& randSeed);
    __host__ __device__ void ResetReservoir();
    __host__ __device__ bool CheckIfValid();
};

#endif
