#ifndef TRIANGLE_H
#define TRIANGLE_H
#include <cstdint>
#include "AABB.cuh"

struct Triangle
{
    uint32_t v0, v1, v2;
    int materialIndex = 0;

    AABB aabb;

    static glm::vec3 GetBarycentricCoords(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2);
    static glm::vec3 GetTriangleNormal(const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2);
    static float GetTriangleArea(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2);
    static float GetTriangleAreaSquared(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2);
};

#endif
