#ifndef COLORUTILS_H
#define COLORUTILS_H
#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#define GLM_FORCE_CUDA
#include <cstdint>
#include <glm/vec4.hpp>

#include "../Classes/BaseClasses/Vector4f.h"

namespace ColorUtils
{
    __host__ __device__ __forceinline__ uint32_t ConvertToRGBA(const glm::vec4& color)
    {
        uint8_t r = (uint8_t)(color.x * 255.0f);
        uint8_t g = (uint8_t)(color.y * 255.0f);
        uint8_t b = (uint8_t)(color.z * 255.0f);
        uint8_t a = (uint8_t)(color.w * 255.0f);

        return (a << 24) | (b << 16) | (g << 8) | r;
    }

    __host__ __device__ __forceinline__ uint32_t ConvertToRGBA(const Vector4f& color)
    {
        uint8_t r = (uint8_t)(color.x * 255.0f);
        uint8_t g = (uint8_t)(color.y * 255.0f);
        uint8_t b = (uint8_t)(color.z * 255.0f);
        uint8_t a = (uint8_t)(color.w * 255.0f);

        return (a << 24) | (b << 16) | (g << 8) | r;
    }

    __host__ __device__ __forceinline__ glm::vec4 UnpackABGR(uint32_t abgr)
    {
        float r = (float)((abgr) & 0xFF) * (1.0f / 255.0f);
        float g = (float)((abgr >> 8) & 0xFF) * (1.0f / 255.0f);
        float b = (float)((abgr >> 16) & 0xFF) * (1.0f / 255.0f);
        float a = (float)((abgr >> 24) & 0xFF) * (1.0f / 255.0f);
        return glm::vec4(r, g, b, a);
    }
}


#endif
