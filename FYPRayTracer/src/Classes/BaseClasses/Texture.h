#ifndef TEXTURE_H
#define TEXTURE_H
#include <string>

struct Texture
{
    uint32_t* pixels = nullptr; //  ABGR format (not RGBA so first 8 bits at the left is alpha channel then subsequently blue, green, and red)
    uint32_t width = 0, height = 0;

    Texture(){}
    Texture(std::string& imageFilePath);
    ~Texture();

    void FreeTexture();
};

#endif
