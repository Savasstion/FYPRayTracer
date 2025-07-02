#include "MisUtils.h"
#include <fstream>
#include <vector>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>


bool MisUtils::SaveABGRToBMP(const std::string& filename, const uint32_t* abgrPixels, int width,
                          int height)
{
    
    // Each row must be padded to a multiple of 4 bytes
    int rowStride = ((width * 3 + 3) / 4) * 4;
    int pixelDataSize = rowStride * height;
    int fileSize = 54 + pixelDataSize;

    // BMP Header (14 bytes)
    uint8_t fileHeader[14] = {
        'B', 'M',                          // Signature
        0, 0, 0, 0,                        // File size
        0, 0,                              // Reserved
        0, 0,                              // Reserved
        54, 0, 0, 0                        // Pixel data offset
    };

    // Fill in file size
    fileHeader[2] = fileSize & 0xFF;
    fileHeader[3] = (fileSize >> 8) & 0xFF;
    fileHeader[4] = (fileSize >> 16) & 0xFF;
    fileHeader[5] = (fileSize >> 24) & 0xFF;

    // DIB Header (40 bytes)
    uint8_t dibHeader[40] = {
        40, 0, 0, 0,                       // Header size
        0, 0, 0, 0,                        // Width
        0, 0, 0, 0,                        // Height
        1, 0,                              // Color planes
        24, 0,                             // Bits per pixel (BGR)
        0, 0, 0, 0,                        // Compression (0 = none)
        0, 0, 0, 0,                        // Image size (can be 0 for no compression)
        0, 0, 0, 0,                        // X pixels per meter
        0, 0, 0, 0,                        // Y pixels per meter
        0, 0, 0, 0,                        // Total colors
        0, 0, 0, 0                         // Important colors
    };

    // Fill in width and height (little endian)
    dibHeader[4]  = width & 0xFF;
    dibHeader[5]  = (width >> 8) & 0xFF;
    dibHeader[6]  = (width >> 16) & 0xFF;
    dibHeader[7]  = (width >> 24) & 0xFF;
    dibHeader[8]  = height & 0xFF;
    dibHeader[9]  = (height >> 8) & 0xFF;
    dibHeader[10] = (height >> 16) & 0xFF;
    dibHeader[11] = (height >> 24) & 0xFF;

    // Open file
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    // Write headers
    file.write(reinterpret_cast<char*>(fileHeader), sizeof(fileHeader));
    file.write(reinterpret_cast<char*>(dibHeader), sizeof(dibHeader));

    // Write pixel data (bottom-up BMP format)
    std::vector<uint8_t> rowBuffer(rowStride);
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            uint32_t abgr = abgrPixels[y * width + x];
            uint8_t a = (abgr >> 24) & 0xFF;
            uint8_t b = (abgr >> 16) & 0xFF;
            uint8_t g = (abgr >> 8) & 0xFF;
            uint8_t r = (abgr >> 0) & 0xFF;

            rowBuffer[x * 3 + 0] = b;
            rowBuffer[x * 3 + 1] = g;
            rowBuffer[x * 3 + 2] = r;
        }

        // Pad row if needed
        for (int p = width * 3; p < rowStride; ++p)
            rowBuffer[p] = 0;

        file.write(reinterpret_cast<char*>(rowBuffer.data()), rowStride);
    }

    file.close();
    return true;
}

std::string MisUtils::GetTimestampedFilename(const std::string& baseName, const std::string& extension)
{
    using namespace std::chrono;

    auto now = system_clock::now();
    std::time_t now_c = system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now_c);
#else
        localtime_r(&now_c, &tm);
#endif

    std::ostringstream oss;
    oss << baseName << "_"
        << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S")
        << extension;

    return oss.str();
}
