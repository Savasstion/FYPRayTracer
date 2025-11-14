#ifndef RESTIR_DI_RESERVOIR_CUH
#define RESTIR_DI_RESERVOIR_CUH
#include <cstdint>
#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include "../Utility/MathUtils.cuh"

struct ReSTIR_DI_Reservoir
{
    uint32_t indexEmissive;             // index of most important light
    float weightEmissive;               // light weight
    float weightSum;                    // sum of all weights for all lights processed
    uint32_t emissiveProcessedCount;       // number of lights processed for this reservoir
    
    __host__ __device__ bool UpdateReservoir(uint32_t candidateEmissiveIndex, float weight, uint32_t& randSeed);
};

#endif