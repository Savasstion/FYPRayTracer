#ifndef TEXTURE_GPU_H
#define TEXTURE_GPU_H
// #include <cuda.h>
// #define GLM_FORCE_CUDA
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <string>
#include "Texture.h"


struct TextureGPU
{
    uint32_t* pixels = nullptr; //  ABGR format (not RGBA so first 8 bits at the left is alpha channel then subsequently blue, green, and red)
    uint32_t width = 0, height = 0;

    __host__ TextureGPU(){}
    __host__ TextureGPU(Texture& cpuTexture);
    __host__ ~TextureGPU();
    __host__ void FreeTextureGPU();
    //  Nearest Neighbour Sampling
    __host__ __device__ uint32_t SampleNearest(float u, float v);
    //  Bilinear Filtering
    __host__ __device__ uint32_t SampleBilinear(float u, float v);
};

#endif