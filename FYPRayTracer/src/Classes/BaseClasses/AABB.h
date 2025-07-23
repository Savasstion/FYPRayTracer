#ifndef AABB_H
#define AABB_H

#include "Vector3f.h"
#include <glm/ext/vector_float2.hpp>
struct AABB
{
    Vector3f lowerBound;    // Bottom-left corner
    Vector3f upperBound;    // Upper-right corner

    AABB(const Vector3f lowerBound, const Vector3f upperBound)
        : lowerBound(lowerBound), upperBound(upperBound) {
    }

    AABB(const glm::vec3 lowerBound, const glm::vec3 upperBound)
        : lowerBound(Vector3f(lowerBound)), upperBound(Vector3f(upperBound)) {
    }

    AABB()
        : lowerBound(Vector3f(0, 0, 0)), upperBound(Vector3f(0, 0, 0)) {
    }
    static AABB UnionAABB(const AABB& a, const AABB& b)
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
    
    static bool isIntersect(const AABB& a, const AABB& b)
    {
        return !(a.upperBound.x < b.lowerBound.x || a.lowerBound.x > b.upperBound.x ||
                 a.upperBound.y < b.lowerBound.y || a.lowerBound.y > b.upperBound.y ||
                 a.upperBound.z < b.lowerBound.z || a.lowerBound.z > b.upperBound.z);
    }

    bool isIntersect(const AABB& other) const
    {
        return AABB::isIntersect(*this, other);
    }
};

#endif