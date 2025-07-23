#ifndef VECTOR3F_H
#define VECTOR3F_H

//  We dont use cmath to prevent dependency issues
#include "Vector2f.h"
#include "../../Utility/MathUtils.h"

class Vector3f {
public:
    float x, y, z;

    // Constructors
    Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}
    Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
    Vector3f(glm::vec3 vector) : x(vector.x), y(vector.y), z(vector.z) {}
    Vector3f(glm::vec2 vector) : x(vector.x), y(vector.y), z(0.0f) {}

    Vector3f Clamped(const float& min, const float& max)
    {
        Vector3f result;

        result.x = (x < min) ? min : (x > max) ? max : x;
        result.y = (y < min) ? min : (y > max) ? max : y;
        result.z = (z < min) ? min : (z > max) ? max : z;

        return result;
    }

    void Clamp(const float& min, const float& max)
    {
        x = (x < min) ? min : (x > max) ? max : x;
        y = (y < min) ? min : (y > max) ? max : y;
        z = (z < min) ? min : (z > max) ? max : z;
        
    }
    
    // Magnitude (length)
    float Magnitude() const {
        return MathUtils::approx_sqrt(x*x + y*y + z*z);    //  square roots are slow so use MagnitudeSquared() when possible
    }

    float MagnitudeSquared() const {
        return x*x + y*y + z*z;
    }

    // Normalize the vector (make it unit length)
    Vector3f Normalized() const {
        float lenSq = MagnitudeSquared();
        if (lenSq > 0.0f) {
            float invMag = MathUtils::fi_sqrt(lenSq * lenSq);
            return Vector3f(x * invMag, y * invMag, z * invMag);
        }
        return Vector3f(0.0f, 0.0f, 0.0f);
    }

    void Normalize() {
        float lenSq = MagnitudeSquared();
        if (lenSq > 0.0f) {
            float invMag = MathUtils::fi_sqrt(lenSq * lenSq);
            x *= invMag;
            y *= invMag;
            z *= invMag;
        }
    }

    // Dot product
    float Dot(const Vector3f& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    float Dot(const Vector2f& other) const {
        return x * other.x + y * other.y; // Treat Vector2f as (x, y, 0)
    }

    static float Dot(const Vector3f& a, const Vector3f& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    static float Dot(const Vector3f& a, const Vector2f& b) {
        return a.x * b.x + a.y * b.y;
    }


    // Cross product
    Vector3f Cross(const Vector3f& other) const {
        return Vector3f(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    Vector3f Cross(const Vector2f& other) const {
        // Treat Vector2f as (x, y, 0) and return the 3D cross product
        return Vector3f(
            0.0f,
            0.0f,
            x * other.y - y * other.x
        );
    }

    static Vector3f Cross(const Vector3f& a, const Vector3f& b) {
        return Vector3f(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    static Vector3f Cross(const Vector3f& a, const Vector2f& b) {
        return Vector3f(
            0.0f,
            0.0f,
            a.x * b.y - a.y * b.x
        );
    }

    Vector3f min(const Vector3f& other) const
    {
        return Vector3f(
            (x < other.x) ? x : other.x,
            (y < other.y) ? y : other.y,
            (z < other.z) ? z : other.z
        );
    }

    Vector3f max(const Vector3f& other) const
    {
        return Vector3f(
            (x > other.x) ? x : other.x,
            (y > other.y) ? y : other.y,
            (z > other.z) ? z : other.z
        );
    }

    // Operator overloading
    Vector3f operator+(const Vector3f& other) const {
        return Vector3f(x + other.x, y + other.y, z + other.z);
    }

    Vector3f operator+(const Vector2f& other) const {
        return Vector3f(x + other.x, y + other.y, z);
    }

    Vector3f operator+(const float& scalar) const {
        return Vector3f(x + scalar, y + scalar, z + scalar);
    }

    Vector3f operator-(const Vector3f& other) const {
        return Vector3f(x - other.x, y - other.y, z - other.z);
    }

    Vector3f operator-(const Vector2f& other) const {
        return Vector3f(x - other.x, y - other.y, z);
    }

    Vector3f operator-(const float& scalar) const {
        return Vector3f(x - scalar, y - scalar, z - scalar);
    }

    Vector3f operator*(float scalar) const {
        return Vector3f(x * scalar, y * scalar, z * scalar);
    }

    Vector3f operator/(float scalar) const {
        return Vector3f(x / scalar, y / scalar, z / scalar);
    }

    Vector3f& operator+=(const Vector3f& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vector3f& operator+=(const Vector2f& other) {
        x += other.x;
        y += other.y;
        return *this;
    }

    Vector3f& operator+=(const float& scalar) {
        x += scalar;
        y += scalar;
        z += scalar;
        return *this;
    }

    Vector3f& operator-=(const Vector3f& other) {
        x -= other.x;
        y -= other.y;
        z -= other.z;
        return *this;
    }

    Vector3f& operator-=(const Vector2f& other) {
        x -= other.x;
        y -= other.y;
        return *this;
    }

    Vector3f& operator-=(const float& scalar) {
        x -= scalar;
        y -= scalar;
        z -= scalar;
        return *this;
    }

    Vector3f& operator*=(float scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vector3f& operator/=(float scalar) {
        x /= scalar;
        y /= scalar;
        z /= scalar;
        return *this;
    }
};

#endif