#ifndef RESTIR_GI_RESERVOIR_CUH
#define RESTIR_GI_RESERVOIR_CUH
#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>

struct ReSTIR_GI_Reservoir
{
    struct PathSample
    {
        glm::vec3 visiblePoint{0.0f};
        glm::vec2 visibleNormal{0.0f};  // Octahedral Encoded
        glm::vec3 samplePoint{0.0f};
        glm::vec2 sampleNormal{0.0f};  // Octahedral Encoded
        glm::vec3 outgoingRadiance{0.0f};   //  Outgoing radiance at sample point in RGB
        uint32_t randSeed = 0;  // Random number used for path
        float samplePDF;

        __host__ __device__ void ResetSample();
    };

    PathSample sample;
    float weightSample = 0.0f;
    uint32_t pathProcessedCount = 0;
    float weightSum = 0.0f;

    __host__ __device__ bool UpdateReservoir(const PathSample& newSample, float newWeight, float pdf, uint32_t& randSeed);
    __host__ __device__ bool UpdateReservoir(const PathSample& newSample, float newWeight, uint32_t count, float pdf, uint32_t& randSeed);
    __host__ __device__ bool MergeReservoir(const ReSTIR_GI_Reservoir& otherReservoir, float pdf, uint32_t& randSeed);
    __host__ __device__ void ResetReservoir();
    __host__ __device__ bool CheckIfValid() const;
};

#endif
