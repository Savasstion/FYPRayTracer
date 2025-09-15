#ifndef MATHUTILS_H
#define MATHUTILS_H
#include <cstdint>
#define GLM_FORCE_CUDA
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>

#include "cuda_runtime.h"
#include <device_launch_parameters.h>


//  in case you dont wanna use CUDA's built-in math functions or just dont want to bother with libraries

namespace MathUtils
{
    static constexpr float pi = 3.1415926535f;
    __host__ __device__ __forceinline__ float minFloat(const float a, const float b) 
    {
        return (a < b) ? a : b;
    }

    __host__ __device__ __forceinline__ float maxFloat(const float a, const float b)
    {
        return (a > b) ? a : b;
    }

    //  Quake 3's fast inverse square root
    __host__ __device__ __forceinline__ float fi_sqrt(float number)
    {
        long i;
        float x2, y;
        const float threehalfs = 1.5F;

        x2 = number * 0.5F;
        y = number;
        i = *(long*)&y;                       // evil floating point bit level hacking
        i = 0x5f3759df - (i >> 1);               // what the fuck?
        y = *(float*)&i;
        y = y * (threehalfs - (x2 * y * y));   // 1st iteration
        //	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

        return y;
    }   
    
    //  quick random number generator
    __host__ __device__ __forceinline__ uint32_t pcg_hash(uint32_t input)
    {
        uint32_t state = input * 747796405u + 2891336453u;
        uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }  

    __host__ __device__ __forceinline__ float randomFloat(uint32_t& seed)
    {
        seed = pcg_hash(seed);  //  in case you wanna call this multiple times, so we gonna overwrite the seed to be reused
        return (float)seed / (float)UINT32_MAX; //  return a random float between 0 to 1
    }

    __host__ __device__ __forceinline__ void BuildOrthonormalBasis(const glm::vec3& n, glm::vec3& tangent, glm::vec3& bitangent)
    {
        // Used to help build a local coordinate system
        if (n.x * n.x > n.z * n.z)
            tangent = glm::normalize(glm::vec3(-n.y, n.x, 0.0f));
        else
            tangent = glm::normalize(glm::vec3(0.0f, -n.z, n.y));

        bitangent = glm::normalize(glm::cross(n, tangent));
    }

    __host__ __device__ __forceinline__ glm::vec3 CosineSampleHemisphere(const glm::vec3& normal, uint32_t& seed)
    {
        //  Focus on sampling diffuse lights
        float u1 = MathUtils::randomFloat(seed);
        float u2 = MathUtils::randomFloat(seed);

        float r = sqrt(u1);
        float theta = 2.0f * pi * u2;

        float x = r * glm::cos(theta);
        float y = r * glm::sin(theta);
        float z = sqrt(glm::max(0.0f, 1.0f - u1));

        // Convert to world space
        glm::vec3 tangent, bitangent;
        MathUtils::BuildOrthonormalBasis(normal, tangent, bitangent);
        return glm::normalize(tangent * x + bitangent * y + normal * z);
    }

    constexpr __host__ __device__ __forceinline__ float CosineHemispherePDF(float cosTheta){ return cosTheta / pi;}

    __host__ __device__ __forceinline__ glm::vec3 UniformSampleHemisphere(const glm::vec3& normal, uint32_t& seed)
    {
        float u1 = MathUtils::randomFloat(seed);
        float u2 = MathUtils::randomFloat(seed);

        float phi = 2.0f * pi * u1;
        float cosTheta = u2;
        float sinTheta = glm::sqrt(1.0f - cosTheta * cosTheta);

        float x = sinTheta * glm::cos(phi);
        float y = sinTheta * glm::sin(phi);
        float z = cosTheta;

        // Convert from local to world space using normal, tangent, and bitangent
        glm::vec3 tangent, bitangent;
        MathUtils::BuildOrthonormalBasis(normal, tangent, bitangent);
        return glm::normalize(tangent * x + bitangent * y + normal * z);
    }

    constexpr __host__ __device__ __forceinline__ float UniformHemispherePDF() {return 1 / (2 * pi);}

    __host__ __device__ __forceinline__ glm::vec3 GGXSampleHemisphere(const glm::vec3& normal, const glm::vec3& viewVector, float roughness, uint32_t& seed, float& outPDF)
    {
        glm::vec3 L;
        glm::vec3 halfVector;

        do
        {
            float u1 = randomFloat(seed);
            float u2 = randomFloat(seed);
            float a  = roughness * roughness;

            // GGX spherical sampling
            float phi      = 2.0f * pi * u1;
            float cosTheta = glm::sqrt((1.0f - u2) /
                                       (1.0f + (a * a - 1.0f) * u2));
            float sinTheta = glm::sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));

            // Half-vector in tangent space
            glm::vec3 h_tangent(sinTheta * glm::cos(phi),
                                sinTheta * glm::sin(phi),
                                cosTheta);

            // Transform to world space
            glm::vec3 tangent, bitangent;
            BuildOrthonormalBasis(normal, tangent, bitangent);
            halfVector = glm::normalize(
                h_tangent.x * tangent +
                h_tangent.y * bitangent +
                h_tangent.z * normal
            );

            // Reflect V about H to get outgoing direction
            L = glm::reflect(-viewVector, halfVector);

        } while (glm::dot(normal, L) <= 0.0f); // reject below the surface

        // ---- PDF computation -----------------------------------------------
        glm::vec3 h = glm::normalize(viewVector + L);
        float NdotH = glm::max(0.0f, glm::dot(normal, h));
        float VdotH = glm::max(0.0f, glm::dot(viewVector, h));
        float alpha = roughness; // if you use alpha=roughness^2 adjust here

        // GGX NDF
        float denom = (NdotH * NdotH) * (alpha * alpha - 1.0f) + 1.0f;
        float D = (alpha * alpha) / (pi * denom * denom);

        // Reflection PDF
        outPDF = (D * NdotH) / (4.0f * VdotH);
        // --------------------------------------------------------------------

        return glm::normalize(L);
    }

    __host__ __device__ __forceinline__ glm::vec3 BRDFSampleHemisphere(const glm::vec3& normal, const glm::vec3& viewingVector, const glm::vec3& albedo, float metallic, float roughness, uint32_t& seed, float& outPDF)
    {
        float u1 = MathUtils::randomFloat(seed);
        float u2 = MathUtils::randomFloat(seed);
        float u3 = MathUtils::randomFloat(seed);
        glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);
        glm::vec3 H, L;
        glm::vec3 T, B;
        BuildOrthonormalBasis(normal, T, B);

        // Schlick approximation for Fresnel
        glm::vec3 F = F0 + (1.0f - F0) * glm::pow(1.0f - glm::max(glm::dot(normal, viewingVector), 0.0f), 5.0f);
        float specularWeight = glm::max(glm::max(F.x, F.y), F.z); // Clamp to prevent total diffuse

        if (u3 < specularWeight)
        {
            // --- Sample GGX specular lobe ---
            float a = roughness * roughness;
            float phi = 2.0f * pi * u1;
            float cosTheta = glm::sqrt((1.0f - u2) / (1.0f + (a * a - 1.0f) * u2));
            float sinTheta = glm::sqrt(glm::max(0.0f, 1.0f - cosTheta * cosTheta));

            glm::vec3 h_tangent = glm::vec3(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta);
            H = glm::normalize(h_tangent.x * T + h_tangent.y * B + h_tangent.z * normal);
            L = glm::reflect(-viewingVector, H);

            if (glm::dot(normal, L) <= 0.0f)
            {
                outPDF = 0.0f;
                return glm::vec3(0.0f);
            }

            float NdotH = glm::max(glm::dot(normal, H), 0.0f);
            float VdotH = glm::max(glm::dot(viewingVector, H), 0.0f);
            float D = (a * a) / (pi * glm::pow((NdotH * NdotH) * (a * a - 1.0f) + 1.0f, 2.0f));
            float pdf = (D * NdotH) / (4.0f * VdotH);

            outPDF = pdf * specularWeight;
        }
        else
        {
            // Sample cosine-weighted diffuse lobe
            float phi = 2.0f * pi * u1;
            float cosTheta = glm::sqrt(1.0f - u2);
            float sinTheta = glm::sqrt(u2);

            glm::vec3 L_tangent = glm::vec3(sinTheta * glm::cos(phi), sinTheta * glm::sin(phi), cosTheta);
            L = glm::normalize(L_tangent.x * T + L_tangent.y * B + L_tangent.z * normal);

            float NdotL = glm::max(glm::dot(normal, L), 0.0f);
            outPDF = (NdotL / pi) * (1.0f - specularWeight);
        }

        return L;   //  sample vector within the hemisphere
    }
}


#endif
