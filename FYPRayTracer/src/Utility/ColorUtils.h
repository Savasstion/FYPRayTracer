#ifndef COLORUTILS_H
#define COLORUTILS_H

#include <cstdint>
#include <glm/vec4.hpp>

#include "../Classes/BaseClasses/Vector4f.h"

namespace ColorUtils
{
    uint32_t ConvertToRGBA(const glm::vec4& color);
    uint32_t ConvertToRGBA(const Vector4f& color);
}


#endif
