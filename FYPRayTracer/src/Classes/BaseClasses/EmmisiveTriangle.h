#ifndef EMMISIVE_TRI_H
#define EMMISIVE_TRI_H
#include <cstdint>
#include "../../Utility/MathUtils.cuh"

struct EmmisiveTriangle
{
    uint32_t triangleIndex;
    static constexpr float theta_e = MathUtils::pi / 2.0f;
    static constexpr float theta_o = 0;
};

#endif
