#include "Texture.h"
#include <iostream>
#include "../../../vendor/stb_image/stb_image.h"

Texture::Texture(std::string& imageFilePath)
{
    int w, h, channels;
    stbi_uc* data = stbi_load(imageFilePath.c_str(), &w, &h, &channels, 4);
    if (!data)
        std::cerr << ("Failed to load image: " + imageFilePath) << std::endl;

    width  = static_cast<uint32_t>(w);
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
}

Texture::~Texture()
{
    FreeTexture();
}

void Texture::FreeTexture()
{
    width = height = 0;
    delete[] pixels;
}
