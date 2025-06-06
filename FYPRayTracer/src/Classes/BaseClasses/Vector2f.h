#ifndef VECTOR2F_H
#define VECTOR2F_H

//  We don't use cmath to prevent dependency issues
#include "../../Utility/MathFunctions.h"

class Vector2f {
public:
    float x, y;

    // Constructors
    Vector2f() : x(0.0f), y(0.0f) {}
    Vector2f(float x, float y) : x(x), y(y) {}

    // Magnitude (length)
    float Magnitude() const {
        return approx_sqrt(x*x + y*y);    // Use MagnitudeSquared() when possible for performance
    }

    float MagnitudeSquared() const {
        return x*x + y*y;
    }

    // Normalize the vector (make it unit length)
    Vector2f Normalized() const {
        float lenSq = MagnitudeSquared();
        if (lenSq > 0.0f) {
            float invMag = fi_sqrt(lenSq * lenSq);
            return Vector2f(x * invMag, y * invMag);
        }
        return Vector2f(0.0f, 0.0f);
    }

    void Normalize() {
        float lenSq = MagnitudeSquared();
        if (lenSq > 0.0f) {
            float invMag = fi_sqrt(lenSq * lenSq);
            x *= invMag;
            y *= invMag;
        }
    }

    // Dot product
    float Dot(const Vector2f& other) const {
        return x * other.x + y * other.y;
    }

    static float Dot(const Vector2f& a, const Vector2f& b) {
        return a.x * b.x + a.y * b.y;
    }

    // Operator overloading
    Vector2f operator+(const Vector2f& other) const {
        return Vector2f(x + other.x, y + other.y);
    }

    Vector2f operator-(const Vector2f& other) const {
        return Vector2f(x - other.x, y - other.y);
    }

    Vector2f operator*(float scalar) const {
        return Vector2f(x * scalar, y * scalar);
    }

    Vector2f operator/(float scalar) const {
        return Vector2f(x / scalar, y / scalar);
    }

    Vector2f& operator+=(const Vector2f& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Vector2f& operator-=(const Vector2f& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Vector2f& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    Vector2f& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        return *this;
    }
};

#endif
