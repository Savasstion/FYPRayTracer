#ifndef CONE_BOUNDS_H
#define CONE_BOUNDS_H
#include <glm/vec3.hpp>

struct ConeBounds
{
    glm::vec3 axis{0.0f};
    float theta_o = 0.0f;
    float theta_e = 0.0f;

    static ConeBounds UnionCone(ConeBounds a, ConeBounds b);
};

#endif