#include "TextureGPU.cuh"
#include <iostream>
#include <glm/vec4.hpp>
#include "../../Utility/ColorUtils.cuh"

TextureGPU::TextureGPU(Texture& cpuTexture)
{
    width = cpuTexture.width;
    height = cpuTexture.height;
    
    cudaError_t err = cudaMalloc(&pixels, sizeof(uint32_t) * width * height);
    if (err != cudaSuccess)
        std::cerr << "cudaMalloc TextureGPU error: " << cudaGetErrorString(err) << std::endl;

    err = cudaMemcpy(pixels, cpuTexture.pixels, width * height * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
        std::cerr << "cudaMemcpy TextureGPU error: " << cudaGetErrorString(err) << std::endl;
}

TextureGPU::~TextureGPU()
{
    FreeTextureGPU();
}

void TextureGPU::FreeTextureGPU()
{
    width = height = 0;
    if (pixels)
        cudaFree(pixels);
    pixels = nullptr;
}

__host__ __device__ uint32_t TextureGPU::SampleNearest(float u, float v)
{
    // u and v should be [0,1] texture coordinates
    uint32_t x = static_cast<uint32_t>(u * (width  - 1));
    uint32_t y = static_cast<uint32_t>(v * (height - 1));

    return pixels[y * width + x];
}

__host__ __device__ uint32_t TextureGPU::SampleBilinear(float u, float v)
{
    // Clamp UV into [0, 1]
    u = (u < 0.0f) ? 0.0f : (u > 1.0f ? 1.0f : u);
    v = (v < 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v);

    // Transform into texture space
    float x = u * (width  - 1);
    float y = v * (height - 1);

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = (x0 + 1 < (int)width)  ? x0 + 1 : x0;
    int y1 = (y0 + 1 < (int)height) ? y0 + 1 : y0;

    float tx = x - (float)x0;
    float ty = y - (float)y0;

    // Fetch the four nearest texels
    uint32_t p00 = pixels[y0 * width + x0];
    uint32_t p10 = pixels[y0 * width + x1];
    uint32_t p01 = pixels[y1 * width + x0];
    uint32_t p11 = pixels[y1 * width + x1];

    // Convert to float colors
    glm::vec4 c00 = ColorUtils::UnpackABGR(p00);
    glm::vec4 c10 = ColorUtils::UnpackABGR(p10);
    glm::vec4 c01 = ColorUtils::UnpackABGR(p01);
    glm::vec4 c11 = ColorUtils::UnpackABGR(p11);

    // Bilinear interpolation
    glm::vec4 cx0 = c00 * (1.0f - tx) + c10 * tx;
    glm::vec4 cx1 = c01 * (1.0f - tx) + c11 * tx;
    glm::vec4 finalColor = cx0 * (1.0f - ty) + cx1 * ty;

    return ColorUtils::ConvertToRGBA(finalColor);
}
