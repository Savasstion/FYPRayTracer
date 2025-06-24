#include "MathUtils.h"
#include <glm/detail/func_geometric.inl>
#include <glm/trigonometric.hpp>

float MathUtils::fi_sqrt(float number)
{
    long i;
    float x2, y;
    const float threehalfs = 1.5F;

    x2 = number * 0.5F;
    y  = number;
    i  = * ( long * ) &y;                       // evil floating point bit level hacking
    i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
    y  = * ( float * ) &i;
    y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
    //	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

    return y;
}

float MathUtils::approx_sqrt(float number) //Newton-Raphson square root
{
    //  std::sqrt is faster so use this when std lib cant be used
    
    if (number <= 0.0f) return 0.0f;

    float x = number;
    float approx = number * 0.5f;
    // Two iterations for reasonable accuracy
    approx = 0.5f * (approx + number / approx);
    approx = 0.5f * (approx + number / approx);
    return approx;
}

uint32_t MathUtils::pcg_hash(uint32_t input)
{
    uint32_t state = input * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float MathUtils::randomFloat(uint32_t& seed)
{
    seed = pcg_hash(seed);  //  in case you wanna call this multiple times, so we gonna overwrite the seed to be reused
    return (float)seed / (float)UINT32_MAX; //  return a random float between 0 to 1
}

void MathUtils::BuildOrthonormalBasis(const glm::vec3& n, glm::vec3& tangent, glm::vec3& bitangent)
{
    if (fabs(n.x) > fabs(n.z))
        tangent = glm::normalize(glm::vec3(-n.y, n.x, 0.0f));
    else
        tangent = glm::normalize(glm::vec3(0.0f, -n.z, n.y));
    bitangent = glm::normalize(glm::cross(n, tangent));
}

glm::vec3 MathUtils::CosineSampleHemisphere(const glm::vec3& normal, uint32_t& seed)
{
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


glm::vec3 MathUtils::UniformSampleHemisphere(const glm::vec3& normal, uint32_t& seed)
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



