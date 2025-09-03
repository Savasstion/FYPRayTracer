#ifndef CONE_BOUNDS_H
#define CONE_BOUNDS_H
#include <glm/vec3.hpp>
#include "AABB.cuh"

struct ConeBounds
{
    glm::vec3 axis{0.0f};
    float theta_o = 0.0f;
    float theta_e = 0.0f;

    static ConeBounds UnionCone(ConeBounds a, ConeBounds b);
    static ConeBounds FindConeThatEnvelopsAABBFromPoint(const AABB& aabb, glm::vec3 pointPos);
};

#endif