#include "Texture.cuh"
#include <iostream>
#include <glm/vec4.hpp>
#include "../../../vendor/stb_image/stb_image.h"
#include "../../Utility/ColorUtils.cuh"
#include <filesystem>

Texture::Texture(std::string& imageFilePath)
{
    int w, h, channels;
    stbi_uc* data = stbi_load(imageFilePath.c_str(), &w, &h, &channels, 4);
    if (!data)
    {
        std::cerr << ("Failed to load image: " + imageFilePath) << std::endl;
        stbi_image_free(data);
        return;
    }
    
    width = static_cast<uint32_t>(w);
    height = static_cast<uint32_t>(h);

    pixels = new uint32_t[width * height];

    for (uint32_t i = 0; i < width * height; i++)
    {
        uint8_t R = data[i * 4 + 0];
        uint8_t G = data[i * 4 + 1];
        uint8_t B = data[i * 4 + 2];
        uint8_t A = data[i * 4 + 3];

        // store in ABGR format (A | B | G | R)
        pixels[i] = (A << 24) | (B << 16) | (G << 8) | (R << 0);
    }

    stbi_image_free(data);

    //  Get actual file name. Example : "BananaDiffuse.png"
    std::filesystem::path p(imageFilePath);
    fileName = p.filename().string();
}

Texture TextureToHostTextureGPU(const Texture& h_tex)
{
    Texture h_gpuTex{};
    cudaError_t err;

    // Basic members copying
    h_gpuTex.height = h_tex.height;
    h_gpuTex.width = h_tex.width;

    // Copy nodes array to device if available
    if (h_gpuTex.height > 0 && h_gpuTex.width > 0)
    {
        uint32_t* d_pixels = nullptr;
        err = cudaMalloc(&d_pixels, h_gpuTex.height * h_gpuTex.width * sizeof(uint32_t));
        if (err != cudaSuccess)
        {
            std::cerr << "cudaMalloc d_pixels error: " << cudaGetErrorString(err) << std::endl;
        }
        else
        {
            err = cudaMemcpy(d_pixels, h_tex.pixels,
                             h_gpuTex.height * h_gpuTex.width * sizeof(uint32_t),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                std::cerr << "cudaMemcpy d_pixels error: " << cudaGetErrorString(err) << std::endl;
            }
        }
        h_gpuTex.pixels = d_pixels;
    }
    else
    {
        h_gpuTex.pixels = nullptr;
    }

    return h_gpuTex;
}

void Texture::FreeTexture()
{
    width = height = 0;
    delete[] pixels;
}

__host__ void Texture::FreeTextureGPU()
{
    width = height = 0;
    if (pixels)
        cudaFree(pixels);
    pixels = nullptr;
}

__host__ __device__ uint32_t Texture::SampleNearest(float u, float v)
{
    // u and v should be [0,1] texture coordinates
    uint32_t x = static_cast<uint32_t>(u * (width - 1));
    uint32_t y = static_cast<uint32_t>(v * (height - 1));

    return pixels[y * width + x];
}

__host__ __device__ uint32_t Texture::SampleBilinear(float u, float v)
{
    // Clamp UV into [0, 1]
    u = (u < 0.0f) ? 0.0f : (u > 1.0f ? 1.0f : u);
    v = (v < 0.0f) ? 0.0f : (v > 1.0f ? 1.0f : v);

    // Transform into texture space
    float x = u * (width - 1);
    float y = v * (height - 1);

    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = (x0 + 1 < (int)width) ? x0 + 1 : x0;
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
