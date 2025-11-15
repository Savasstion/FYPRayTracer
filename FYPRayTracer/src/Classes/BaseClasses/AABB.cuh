#ifndef AABB_H
#define AABB_H
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#define GLM_FORCE_CUDA
#include "Vector3f.cuh"
#include <glm/ext/vector_float2.hpp>

struct AABB
{
    Vector3f lowerBound; // Bottom-left corner
    Vector3f upperBound; // Upper-right corner
    Vector3f centroidPos;

    __host__ __device__ AABB()
        : lowerBound(Vector3f(0, 0, 0)),
          upperBound(Vector3f(0, 0, 0)),
          centroidPos(Vector3f(0, 0, 0))
    {
    }

    __host__ __device__ AABB(const Vector3f lowerBound, const Vector3f upperBound)
        : lowerBound(lowerBound),
          upperBound(upperBound),
          centroidPos((lowerBound + upperBound) * 0.5f)
    {
    }

    __host__ __device__ AABB(const glm::vec3 lowerBound, const glm::vec3 upperBound)
        : lowerBound(Vector3f(lowerBound)),
          upperBound(Vector3f(upperBound)),
          centroidPos((Vector3f(lowerBound) + Vector3f(upperBound)) * 0.5f)
    {
    }

    __host__ __device__ __forceinline__ static Vector3f FindCentroid(AABB& aabb)
    {
        aabb.centroidPos = (Vector3f(aabb.lowerBound) + Vector3f(aabb.upperBound)) * 0.5f;
        return aabb.centroidPos;
    }

    __host__ __device__ __forceinline__ static AABB UnionAABB(const AABB& a, const AABB& b)
    {
        AABB c;
        c.lowerBound = Vector3f(
            MathUtils::minFloat(a.lowerBound.x, b.lowerBound.x),
            MathUtils::minFloat(a.lowerBound.y, b.lowerBound.y),
            MathUtils::minFloat(a.lowerBound.z, b.lowerBound.z)
        );
        c.upperBound = Vector3f(
            MathUtils::maxFloat(a.upperBound.x, b.upperBound.x),
            MathUtils::maxFloat(a.upperBound.y, b.upperBound.y),
            MathUtils::maxFloat(a.upperBound.z, b.upperBound.z)
        );
        return c;
    }

    __host__ __device__ __forceinline__ static bool isIntersect(const AABB& a, const AABB& b)
    {
        return !(a.upperBound.x < b.lowerBound.x || a.lowerBound.x > b.upperBound.x ||
            a.upperBound.y < b.lowerBound.y || a.lowerBound.y > b.upperBound.y ||
            a.upperBound.z < b.lowerBound.z || a.lowerBound.z > b.upperBound.z);
    }

    __host__ __device__ __forceinline__ bool isIntersect(const AABB& other) const
    {
        return AABB::isIntersect(*this, other);
    }

    __host__ __device__ __forceinline__ float GetSurfaceArea() const
    {
        float dx = upperBound.x - lowerBound.x;
        float dy = upperBound.y - lowerBound.y;
        float dz = upperBound.z - lowerBound.z;
        return 2.0f * (dx * dy + dy * dz + dz * dx);
    }
};

#endif
