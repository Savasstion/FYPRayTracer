#ifndef VECTOR4F_H
#define VECTOR4F_H

// We don't use cmath to prevent dependency issues
#include "Vector3f.cuh"
#include "../../Utility/MathUtils.h"

class Vector4f {
public:
    float x, y, z, w;

    // Constructors
    Vector4f() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
    Vector4f(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

    Vector4f Clamped(const float& min, const float& max)
    {
        Vector4f result;

        result.x = (x < min) ? min : (x > max) ? max : x;
        result.y = (y < min) ? min : (y > max) ? max : y;
        result.z = (z < min) ? min : (z > max) ? max : z;
        result.w = (w < min) ? min : (w > max) ? max : w;

        return result;
    }

    void Clamp(const float& min, const float& max)
    {
        x = (x < min) ? min : (x > max) ? max : x;
        y = (y < min) ? min : (y > max) ? max : y;
        z = (z < min) ? min : (z > max) ? max : z;
        w = (w < min) ? min : (w > max) ? max : w;
        
    }
    
    // Magnitude (length)
    float Magnitude() const {
        return MathUtils::approx_sqrt(x * x + y * y + z * z + w * w); // Use MagnitudeSquared() when possible for performance
    }

    float MagnitudeSquared() const {
        return x * x + y * y + z * z + w * w;
    }

    // Normalize the vector (make it unit length)
    Vector4f Normalized() const {
        float lenSq = MagnitudeSquared();
        if (lenSq > 0.0f) {
            float invMag = MathUtils::fi_sqrt(lenSq * lenSq);
            return Vector4f(x * invMag, y * invMag, z * invMag, w * invMag);
        }
        return Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
    }

    void Normalize() {
        float lenSq = MagnitudeSquared();
        if (lenSq > 0.0f) {
            float invMag = MathUtils::fi_sqrt(lenSq * lenSq);
            x *= invMag;
            y *= invMag;
            z *= invMag;
            w *= invMag;
        }
    }

    // Dot product
    float Dot(const Vector4f& other) const {
        return x * other.x + y * other.y + z * other.z + w * other.w;
    }

    float Dot(const Vector3f& other) const {
        // Treat Vector3f as (x,y,z,0)
        return x * other.x + y * other.y + z * other.z;
    }

    static float Dot(const Vector4f& a, const Vector4f& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }

    // Operator overloading
    Vector4f operator+(const Vector4f& other) const {
        return Vector4f(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    Vector4f operator+(const Vector3f& other) const {
        return Vector4f(x + other.x, y + other.y, z + other.z, w);
    }

    Vector4f operator+(const Vector2f& other) const {
        return Vector4f(x + other.x, y + other.y, z, w);
    }
    
    Vector4f operator+(const float& scalar) const {
        return Vector4f(x + scalar, y + scalar, z + scalar, w + scalar);
    }

    Vector4f operator-(const Vector4f& other) const {
        return Vector4f(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    Vector4f operator-(const Vector3f& other) const {
        return Vector4f(x - other.x, y - other.y, z - other.z, w);
    }

    Vector4f operator-(const Vector2f& other) const {
        return Vector4f(x - other.x, y - other.y, z, w);
    }

    Vector4f operator-(const float& scalar) const {
        return Vector4f(x - scalar, y - scalar, z - scalar, w - scalar);
    }

    Vector4f operator*(float scalar) const {
        return Vector4f(x * scalar, y * scalar, z * scalar, w * scalar);
    }

    Vector4f operator/(float scalar) const {
        return Vector4f(x / scalar, y / scalar, z / scalar, w / scalar);
    }

    Vector4f& operator+=(const Vector4f& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        w += other.w;
        return *this;
    }

    Vector4f& operator+=(const Vector3f& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vector4f& operator+=(const Vector2f& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Vector4f& operator+=(const float& scalar) {
        x += scalar;
        y += scalar;
        z += scalar;
        w += scalar;
        return *this;
    }

    Vector4f& operator-=(const Vector4f& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        w -= other.w;
        return *this;
    }

    Vector4f& operator-=(const Vector3f& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vector4f& operator-=(const Vector2f& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Vector4f& operator-=(const float& scalar) {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        w -= scalar;
        return *this;
    }

    Vector4f& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        w *= scalar;
        return *this;
    }

    Vector4f& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        w /= scalar;
        return *this;
    }
};

#endif
