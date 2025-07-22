#ifndef AABB_H
#define AABB_H

#include "Vector2f.h"
#include <glm/ext/vector_float2.hpp>
struct AABB
{
    Vector2f lowerBound;    // Bottom-left corner
    Vector2f upperBound;    // Upper-right corner

    AABB(const Vector2f lowerBound, const Vector2f upperBound)
        : lowerBound(lowerBound), upperBound(upperBound) {
    }

    AABB(const glm::vec2 lowerBound, const glm::vec2 upperBound)
        : lowerBound(Vector2f(lowerBound)), upperBound(Vector2f(upperBound)) {
    }

    AABB()
        : lowerBound(Vector2f(0, 0)), upperBound(Vector2f(0, 0)) {
    }
    static AABB UnionAABB(const AABB& a, const AABB& b)
    {
        AABB C;
        C.lowerBound = Vector2f(MathUtils::minFloat(a.lowerBound.x, b.lowerBound.x),
            MathUtils::minFloat(a.lowerBound.y, b.lowerBound.y));
        C.upperBound = Vector2f(MathUtils::maxFloat(a.upperBound.x, b.upperBound.x),
            MathUtils::maxFloat(a.upperBound.y, b.upperBound.y));
        return C;
    }
    static bool isIntersect(const AABB& a, const AABB& b)
    {
        return !(a.upperBound.x < b.lowerBound.x ||
            a.lowerBound.x > b.upperBound.x ||
            a.upperBound.y < b.lowerBound.y ||
            a.lowerBound.y > b.upperBound.y);
    }
    bool isIntersect(const AABB& other)
    {
        return AABB::isIntersect(*this, other);
    }
};

#endif