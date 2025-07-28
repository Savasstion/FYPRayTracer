#ifndef VECTOR3F_H
#define VECTOR3F_H

//  Force GLM to be CUDA-compatible
#define GLM_FORCE_CUDA
#include <glm/ext/vector_float3.hpp>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include "Vector2f.h"
#include "../../Utility/MathUtils.cuh"

class Vector3f {
public:
    float x, y, z;

    // Constructors
    __host__ __device__ __forceinline__ Vector3f() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ __forceinline__ Vector3f(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ __forceinline__ Vector3f(glm::vec3 vector) : x(vector.x), y(vector.y), z(vector.z) {}
    __host__ __device__ __forceinline__ Vector3f(glm::vec2 vector) : x(vector.x), y(vector.y), z(0.0f) {}

    __host__ __device__ __forceinline__ Vector3f Clamped(const float& min, const float& max) const {
        Vector3f result;
        result.x = (x < min) ? min : (x > max) ? max : x;
        result.y = (y < min) ? min : (y > max) ? max : y;
        result.z = (z < min) ? min : (z > max) ? max : z;
        return result;
    }

    __host__ __device__ __forceinline__ void Clamp(const float& min, const float& max) {
        x = (x < min) ? min : (x > max) ? max : x;
        y = (y < min) ? min : (y > max) ? max : y;
        z = (z < min) ? min : (z > max) ? max : z;
    }

    __host__ __device__ __forceinline__ float Magnitude() const {
        return MathUtils::approx_sqrt(x * x + y * y + z * z);
    }

    __host__ __device__ __forceinline__ float MagnitudeSquared() const {
        return x * x + y * y + z * z;
    }

    __host__ __device__ __forceinline__ Vector3f Normalized() const {
        float lenSq = MagnitudeSquared();
        if (lenSq > 0.0f) {
            float invMag = MathUtils::fi_sqrt(lenSq * lenSq);
            return Vector3f(x * invMag, y * invMag, z * invMag);
        }
        return Vector3f(0.0f, 0.0f, 0.0f);
    }

    __host__ __device__ __forceinline__ void Normalize() {
        float lenSq = MagnitudeSquared();
        if (lenSq > 0.0f) {
            float invMag = MathUtils::fi_sqrt(lenSq * lenSq);
            x *= invMag;
            y *= invMag;
            z *= invMag;
        }
    }

    __host__ __device__ __forceinline__ float Dot(const Vector3f& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    __host__ __device__ __forceinline__ float Dot(const Vector2f& other) const {
        return x * other.x + y * other.y;
    }

    __host__ __device__ __forceinline__ static float Dot(const Vector3f& a, const Vector3f& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __host__ __device__ __forceinline__ static float Dot(const Vector3f& a, const Vector2f& b) {
        return a.x * b.x + a.y * b.y;
    }

    __host__ __device__ __forceinline__ Vector3f Cross(const Vector3f& other) const {
        return Vector3f(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }

    __host__ __device__ __forceinline__ Vector3f Cross(const Vector2f& other) const {
        return Vector3f(
            0.0f,
            0.0f,
            x * other.y - y * other.x
        );
    }

    __host__ __device__ __forceinline__ static Vector3f Cross(const Vector3f& a, const Vector3f& b) {
        return Vector3f(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }

    __host__ __device__ __forceinline__ static Vector3f Cross(const Vector3f& a, const Vector2f& b) {
        return Vector3f(
            0.0f,
            0.0f,
            a.x * b.y - a.y * b.x
        );
    }

    __host__ __device__ __forceinline__ Vector3f min(const Vector3f& other) const {
        return Vector3f(
            (x < other.x) ? x : other.x,
            (y < other.y) ? y : other.y,
            (z < other.z) ? z : other.z
        );
    }

    __host__ __device__ __forceinline__ Vector3f max(const Vector3f& other) const {
        return Vector3f(
            (x > other.x) ? x : other.x,
            (y > other.y) ? y : other.y,
            (z > other.z) ? z : other.z
        );
    }

    // Operators
    __host__ __device__ __forceinline__ Vector3f operator+(const Vector3f& other) const { return Vector3f(x + other.x, y + other.y, z + other.z); }
    __host__ __device__ __forceinline__ Vector3f operator+(const Vector2f& other) const { return Vector3f(x + other.x, y + other.y, z); }
    __host__ __device__ __forceinline__ Vector3f operator+(const float& scalar) const { return Vector3f(x + scalar, y + scalar, z + scalar); }

    __host__ __device__ __forceinline__ Vector3f operator-(const Vector3f& other) const { return Vector3f(x - other.x, y - other.y, z - other.z); }
    __host__ __device__ __forceinline__ Vector3f operator-(const Vector2f& other) const { return Vector3f(x - other.x, y - other.y, z); }
    __host__ __device__ __forceinline__ Vector3f operator-(const float& scalar) const { return Vector3f(x - scalar, y - scalar, z - scalar); }
__forceinline__ 
    __host__ __device__ __forceinline__ Vector3f operator*(float scalar) const { return Vector3f(x * scalar, y * scalar, z * scalar); }
    __host__ __device__ __forceinline__ Vector3f operator/(float scalar) const { return Vector3f(x / scalar, y / scalar, z / scalar); }
__forceinline__ 
    __host__ __device__ __forceinline__ Vector3f& operator+=(const Vector3f& other) { x += other.x; y += other.y; z += other.z; return *this; }
    __host__ __device__ __forceinline__ Vector3f& operator+=(const Vector2f& other) { x += other.x; y += other.y; return *this; }
    __host__ __device__ __forceinline__ Vector3f& operator+=(const float& scalar) { x += scalar; y += scalar; z += scalar; return *this; }
__forceinline__ 
    __host__ __device__ __forceinline__ Vector3f& operator-=(const Vector3f& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
    __host__ __device__ __forceinline__ Vector3f& operator-=(const Vector2f& other) { x -= other.x; y -= other.y; return *this; }
    __host__ __device__ __forceinline__ Vector3f& operator-=(const float& scalar) { x -= scalar; y -= scalar; z -= scalar; return *this; }
__forceinline__ 
    __host__ __device__ __forceinline__ Vector3f& operator*=(float scalar) { x *= scalar; y *= scalar; z *= scalar; return *this; }
    __host__ __device__ __forceinline__ Vector3f& operator/=(float scalar) { x /= scalar; y /= scalar; z /= scalar; return *this; }

    __host__ __device__ __forceinline__ float operator[](int i) const {
        return (i == 0) ? x : (i == 1) ? y : z;
    }

    __host__ __device__ __forceinline__ float& operator[](int i) {
        return (i == 0) ? x : (i == 1) ? y : z;
    }
};

#endif
