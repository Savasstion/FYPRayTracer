#ifndef TRIANGLE_H
#define TRIANGLE_H
#include <cstdint>

struct Triangle
{
    uint32_t v0, v1, v2;
    int materialIndex = 0;
};

#endif