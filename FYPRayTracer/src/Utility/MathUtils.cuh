#ifndef MATHUTILS_H
#define MATHUTILS_H
#include <cstdint>
#define GLM_FORCE_CUDA
#define GLM_FORCE_INLINE
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec3.hpp>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <glm/gtc/matrix_transform.hpp>

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
        i = *(long*)&y; // evil floating point bit level hacking
        i = 0x5f3759df - (i >> 1); // what the fuck?
        y = *(float*)&i;
        y = y * (threehalfs - (x2 * y * y)); // 1st iteration
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
        seed = pcg_hash(seed);
        //  in case you wanna call this multiple times, so we gonna overwrite the seed to be reused
        return (float)seed / (float)UINT32_MAX; //  return a random float between 0 to 1
    }

    __host__ __device__ __forceinline__ void BuildOrthonormalBasis(const glm::vec3& n, glm::vec3& tangent,
                                                                   glm::vec3& bitangent)
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

        float r = sqrtf(u1);
        float theta = 2.0f * pi * u2;

        float x = r * glm::cos(theta);
        float y = r * glm::sin(theta);
        float z = sqrtf(glm::max(0.0f, 1.0f - u1));

        // Convert to world space
        glm::vec3 tangent, bitangent;
        MathUtils::BuildOrthonormalBasis(normal, tangent, bitangent);
        return glm::normalize(tangent * x + bitangent * y + normal * z);
    }

    constexpr __host__ __device__ __forceinline__ float CosineHemispherePDF(const float& cosTheta)
    {
        return cosTheta / pi;
    }

    __host__ __device__ __forceinline__ glm::vec3 UniformSampleHemisphere(const glm::vec3& normal, uint32_t& seed)
    {
        float u1 = MathUtils::randomFloat(seed);
        float u2 = MathUtils::randomFloat(seed);

        float phi = 2.0f * pi * u1;
        float cosTheta = u2;
        float sinTheta = sqrtf(1.0f - cosTheta * cosTheta);

        float x = sinTheta * glm::cos(phi);
        float y = sinTheta * glm::sin(phi);
        float z = cosTheta;

        // Convert from local to world space using normal, tangent, and bitangent
        glm::vec3 tangent, bitangent;
        MathUtils::BuildOrthonormalBasis(normal, tangent, bitangent);
        return glm::normalize(tangent * x + bitangent * y + normal * z);
    }

    constexpr __host__ __device__ __forceinline__ float UniformHemispherePDF() { return 1 / (2 * pi); }

    __host__ __device__ __forceinline__ glm::vec3 GGXSampleHemisphere(const glm::vec3& normal,
                                                                      const glm::vec3& viewVector, float roughness,
                                                                      uint32_t& seed, float& outPDF)
    {
        // 1) RNG
        float u1 = MathUtils::randomFloat(seed); // implement or replace with your RNG
        float u2 = MathUtils::randomFloat(seed);

        // 2) convert roughness to alpha convention
        float alpha = roughness * roughness; // artist-friendly mapping

        // 3) sample spherical coordinates for half-vector H (Walter / Heitz sampling form)
        float phi = 2.0f * pi * u2;

        // stable expression for cos(theta) when alpha = roughness^2 and using a2 = alpha*alpha
        float cosTheta = sqrtf((1.0f - u1) / (1.0f + (alpha * alpha - 1.0f) * u1));
        cosTheta = glm::clamp(cosTheta, 0.0f, 1.0f);
        float sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));

        // half-vector in tangent space (Ht)
        auto Ht = glm::vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

        // 4) build tangent/bitangent and transform to world space
        glm::vec3 T, B;
        BuildOrthonormalBasis(normal, T, B);
        glm::vec3 H = glm::normalize(Ht.x * T + Ht.y * B + Ht.z * normal);

        // 5) reflect view vector about H to get outgoing L
        glm::vec3 L = glm::reflect(-viewVector, H); // reflect assumes incoming is -V

        // 6) validate hemisphere and compute PDF
        float NdotL = glm::dot(normal, L);
        if (NdotL <= 0.0f)
        {
            outPDF = 0.0f;
            return glm::vec3(0.0f);
        }

        float NdotH = glm::dot(normal, H);
        float VdotH = glm::dot(viewVector, H);
        if (VdotH <= 0.0f || NdotH <= 0.0f)
        {
            outPDF = 0.0f;
            return glm::vec3(0.0f);
        }

        // PDF for H (solid angle) = D(N,H) * (N·H)
        float a2 = alpha * alpha;
        float denom = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
        float D = (a2) / (pi * denom * denom);
        float p_H = D * NdotH;

        // convert to PDF for L using Jacobian of reflection: p(L) = p(H) / (4 * V·H)
        outPDF = p_H / (4.0f * VdotH);

        return L;
    }

    __host__ __device__ __forceinline__ float GGXHemispherePDF(const glm::vec3& N, const glm::vec3& V,
                                                               const glm::vec3& L, float roughness)
    {
        glm::vec3 H = glm::normalize(V + L);
        float NdotH = glm::max(glm::dot(N, H), 0.0f);
        float VdotH = glm::max(glm::dot(V, H), 0.0f);
        if (NdotH <= 0.0f || VdotH <= 0.0f)
            return 0.0f;

        float alpha = roughness * roughness;
        float a2 = alpha * alpha;
        float denom = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
        float D = a2 / (pi * denom * denom);
        return D * NdotH / (4.0f * VdotH);
    }

    __host__ __device__ __forceinline__ glm::vec3 BRDFSampleHemisphere(const glm::vec3& normal,
                                                                       const glm::vec3& viewingVector,
                                                                       const glm::vec3& albedo, float metallic,
                                                                       float roughness, uint32_t& seed, float& outPDF)
    {
        glm::vec3 L;
        float pdfSpecular = 0.0f, pdfDiffuse = 0.0f;

        // --- choose branch ---
        float wSpecular;
        if (metallic == 1.0f)
        {
            return GGXSampleHemisphere(normal, viewingVector, roughness, seed, outPDF);
        }
        else if (metallic == 0.0f)
        {
            L = CosineSampleHemisphere(normal, seed);
            float cosTheta = glm::max(glm::dot(normal, L), 0.0f);
            outPDF = CosineHemispherePDF(cosTheta);
            return L;
        }
        else
        {
            //  Fresnel
            glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);
            glm::vec3 F = F0 + (1.0f - F0) * glm::pow(1.0f - glm::max(glm::dot(normal, viewingVector), 0.0f), 5.0f);
            wSpecular = (F.x + F.y + F.z) / 3.0f;
        }

        float rand = randomFloat(seed);

        if (rand <= wSpecular)
        {
            // specular sample
            L = GGXSampleHemisphere(normal, viewingVector, roughness, seed, pdfSpecular);
            // we still need the diffuse pdf for this same L:
            float cosTheta = glm::max(glm::dot(normal, L), 0.0f);
            pdfDiffuse = CosineHemispherePDF(cosTheta);
        }
        else
        {
            // diffuse sample
            L = CosineSampleHemisphere(normal, seed);
            float cosTheta = glm::max(glm::dot(normal, L), 0.0f);
            pdfDiffuse = CosineHemispherePDF(cosTheta);
            // need GGX pdf for same L:
            pdfSpecular = GGXHemispherePDF(normal, viewingVector, L, roughness); // or your GGX pdf function
        }

        // final mixture pdf
        outPDF = wSpecular * pdfSpecular + (1.0f - wSpecular) * pdfDiffuse;
        return L;
    }

    __host__ __device__ __forceinline__ float BRDFHemispherePDF(const glm::vec3& N, const glm::vec3& V,
                                                                const glm::vec3& L, const glm::vec3& albedo,
                                                                float metallic, float roughness)
    {
        // perfect‐metal or perfect‐dielectric cases
        if (metallic == 1.0f)
        {
            return GGXHemispherePDF(N, V, L, roughness);
        }
        if (metallic == 0.0f)
        {
            float cosTheta = glm::max(glm::dot(N, L), 0.0f);
            return CosineHemispherePDF(cosTheta);
        }

        // Fresnel term to decide specular weight
        glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);
        float NdotV = glm::max(glm::dot(N, V), 0.0f);
        glm::vec3 F = F0 + (1.0f - F0) * glm::pow(1.0f - NdotV, 5.0f);
        float wSpec = (F.x + F.y + F.z) * (1.0f / 3.0f); // average the RGB

        // individual PDFs
        float pdfSpec = GGXHemispherePDF(N, V, L, roughness);
        float cosTheta = glm::max(glm::dot(N, L), 0.0f);
        float pdfDiff = CosineHemispherePDF(cosTheta);

        // mixture PDF
        return wSpec * pdfSpec + (1.0f - wSpec) * pdfDiff;
    }

    __host__ __device__ __forceinline__ glm::vec3 CalculateBRDF(const glm::vec3& N, const glm::vec3& V,
                                                                const glm::vec3& L, const glm::vec3& albedo,
                                                                float metallic, float roughness)
    {
        constexpr float invPI = 1.0f / pi;
        float a = roughness * roughness;
        float a2 = a * a;

        glm::vec3 H = glm::normalize(V + L);
        float NdotL = glm::max(glm::dot(N, L), 0.0f);
        float NdotV = glm::max(glm::dot(N, V), 0.0f);
        float NdotH = glm::max(glm::dot(N, H), 0.0f);
        float VdotH = glm::max(glm::dot(V, H), 0.0f);

        if (NdotL == 0.0f || NdotV == 0.0f)
            return glm::vec3(0.0f);

        // Fresnel (Schlick's approximation)
        glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);
        glm::vec3 F = F0 + (1.0f - F0) * glm::pow(1.0f - VdotH, 5.0f);

        // Geometry Shadowing (Smith)
        float k = roughness / 2.0f;
        float G_V = NdotV / (NdotV * (1.0f - k) + k);
        float G_L = NdotL / (NdotL * (1.0f - k) + k);
        float G = G_V * G_L;

        // Lambertian diffuse
        glm::vec3 kD = (1.0f - F);
        glm::vec3 diffuse = kD * albedo * invPI;

        // Normal Distribution (GGX / Trowbridge-Reitz)
        float denominator = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
        float D = a2 * invPI / glm::max((denominator * denominator), 1e-12f);

        //  Cook-Torrance specular
        glm::vec3 specular = (D * G * F) / glm::max((4.0f * NdotV * NdotL), 1e-12f);

        //diffuse + specular should be max 1, if its above 1 then more energy is created than it should conserve
        //return glm::min(diffuse + specular, glm::vec3(1.0f));
        return diffuse + specular;
    }

    __host__ __device__ __forceinline__ float LinearizeDepth(float depth, float near, float far)
    {
        float z = depth * 2.0 - 1.0; // depth (0..1) to NDC (-1..1)
        float linerizedDepth = (2.0 * near * far) / (far + near - z * (far - near));
        
        return (linerizedDepth * 0.5f) + 0.5f;    //  remapped from -1...1 to 0...1
        
    }

     __host__ __device__ __forceinline__ glm::vec2 EncodeOctahedral(glm::vec3 v)
    {
        v /= (fabsf(v.x) + fabsf(v.y) + fabsf(v.z));

        glm::vec2 enc(v.x, v.y);

        if (v.z < 0.0f)
        {
            float ex = enc.x;
            float ey = enc.y;

            // (1 - abs(v.yx))
            float xx = 1.0f - fabsf(ey);
            float yy = 1.0f - fabsf(ex);

            // signs based on original enc.x, enc.y
            float sx = (ex >= 0.0f) ? 1.0f : -1.0f;
            float sy = (ey >= 0.0f) ? 1.0f : -1.0f;

            enc.x = xx * sx;
            enc.y = yy * sy;
        }

        return enc;
    }

    __host__ __device__ __forceinline__ glm::vec3 DecodeOctahedral(glm::vec2 e)
    {
        float ex = e.x;
        float ey = e.y;

        glm::vec3 v(ex, ey, 1.0f - fabsf(ex) - fabsf(ey));

        if (v.z < 0.0f)
        {
            float sx = (ex >= 0.0f) ? 1.0f : -1.0f;
            float sy = (ey >= 0.0f) ? 1.0f : -1.0f;

            float newX = (1.0f - fabsf(ey)) * sx;
            float newY = (1.0f - fabsf(ex)) * sy;

            v.x = newX;
            v.y = newY;
        }

        return glm::normalize(v);
    }

    __host__ __device__ __forceinline__ glm::vec2 GetNormalizedDeviceCoords(const glm::mat4& projection, const glm::mat4& view, const glm::vec3& worldPos)
    {
        // Calc Clip Space
        glm::vec4 clip = projection * view * glm::vec4(worldPos, 1.0f);

        // Avoid divide zero
        if (clip.w == 0.0f)
            return {0.0f, 0.0f};

        // divide x,y,z by w
        glm::vec3 ndc = glm::vec3(clip) / clip.w;

        // 4. Return only xy
        return {ndc.x, ndc.y};
    }

    __host__ __device__ __forceinline__ glm::vec2 GetUVFromNDC(const glm::mat4& projection, const glm::mat4& view, const glm::vec3& worldPos)
    {
        //  -1...1
        return GetNormalizedDeviceCoords(projection, view, worldPos) * 0.5f + 0.5f;
    }

    __host__ __device__ __forceinline__ glm::vec2 GetUVFromNDC(const glm::vec2& ndc)
    {
        //  0...1
        return ndc * 0.5f + 0.5f;
    }
    
}


#endif
