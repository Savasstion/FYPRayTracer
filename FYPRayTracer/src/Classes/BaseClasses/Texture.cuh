#ifndef TEXTURE_CUH
#define TEXTURE_CUH
#include <string>
// #include <cuda.h>
// #define GLM_FORCE_CUDA
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

struct Texture
{
    uint32_t* pixels = nullptr; //  ABGR format (not RGBA so first 8 bits at the left is alpha channel then subsequently blue, green, and red)
    uint32_t width = 0, height = 0;

    Texture(){}
    Texture(std::string& imageFilePath);
    
    void FreeTexture();
    __host__ void FreeTextureGPU();
    //  Nearest Neighbour Sampling
    __host__ __device__ uint32_t SampleNearest(float u, float v);
    //  Bilinear Filtering
    __host__ __device__ uint32_t SampleBilinear(float u, float v);
};

Texture TextureToHostTextureGPU(const Texture& h_tex);

#endif
