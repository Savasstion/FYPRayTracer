#ifndef MISUTILS_H
#define MISUTILS_H
#include <cstdint>
#include <string>

namespace MisUtils
{
    bool SaveABGRToBMP(const std::string& filename, const uint32_t* abgrPixels, int width, int height);
    std::string GetTimestampedFilename(const std::string& baseName, const std::string& extension = ".bmp");
}

#endif
