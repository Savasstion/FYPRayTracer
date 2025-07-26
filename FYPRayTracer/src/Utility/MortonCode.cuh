#ifndef MORTON_CODE_H
#define MORTON_CODE_H

#include "MathUtils.cuh"
#include "../Classes/BaseClasses/Scene.h"
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

//Expands a 10-bit integer into 30 bits
//by inserting 2 zeros after each bit.
__host__ __device__ __forceinline__ int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
//given 3D point located within the unit square [0,1].
__host__ __device__ __forceinline__ unsigned int morton3D(float x, float y, float z)
{
    //  convert to range of (0,1)
    x = (x - SceneSettings::minSceneBound.x) / (SceneSettings::maxSceneBound.x - SceneSettings::minSceneBound.x);
    y = (y - SceneSettings::minSceneBound.y) / (SceneSettings::maxSceneBound.y - SceneSettings::minSceneBound.y);
    z = (z - SceneSettings::minSceneBound.z) / (SceneSettings::maxSceneBound.z - SceneSettings::minSceneBound.z);

    
    x = MathUtils::minFloat(MathUtils::maxFloat(x * 1024.0f, 0.0f), 1023.0f);
    y = MathUtils::minFloat(MathUtils::maxFloat(y * 1024.0f, 0.0f), 1023.0f);
    z = MathUtils::minFloat(MathUtils::maxFloat(z * 1024.0f, 0.0f), 1023.0f);
    
    unsigned int xx = expandBits((unsigned int)x);
    unsigned int yy = expandBits((unsigned int)y);
    unsigned int zz = expandBits((unsigned int)z);
    return xx * 4 + yy * 2 + zz;
    
}

#endif