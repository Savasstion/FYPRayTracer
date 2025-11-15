#ifndef TRIANGLE_H
#define TRIANGLE_H
#include <cstdint>
#include "AABB.cuh"
#include <glm/gtx/norm.hpp>

struct Triangle
{
    uint32_t v0, v1, v2;
    int materialIndex = 0;

    AABB aabb;

    __host__ __device__ __forceinline__ static glm::vec3 GetBarycentricCoords(
        const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
    {
        return (p0 + p1 + p2) / 3.0f;
    }

    __host__ __device__ __forceinline__ static glm::vec3 GetRandomPointOnTriangle(
        const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, uint32_t& seed)
    {
        float r1 = MathUtils::randomFloat(seed);
        float r2 = MathUtils::randomFloat(seed);

        // Warp r1 for uniform area sampling
        float sqrtR1 = sqrtf(r1);

        float u = 1.0f - sqrtR1;
        float v = (1.0f - r2) * sqrtR1;
        float w = r2 * sqrtR1;

        return u * p0 + v * p1 + w * p2;
    }

    __host__ __device__ __forceinline__ static glm::vec3 GetTriangleNormal(
        const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2)
    {
        // Average the vertex normals
        glm::vec3 normal = (n0 + n1 + n2) / 3.0f;

        return glm::normalize(normal);
    }

    __host__ __device__ __forceinline__ static float GetTriangleArea(const glm::vec3& p0, const glm::vec3& p1,
                                                                     const glm::vec3& p2)
    {
        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        return 0.5f * glm::length(glm::cross(edge1, edge2));
    }

    __host__ __device__ __forceinline__ static float GetTriangleAreaSquared(
        const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
    {
        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        return 0.25f * glm::length2(glm::cross(edge1, edge2)); // (area^2)
    }
};

#endif
