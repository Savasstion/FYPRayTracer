#ifndef MATHUTILS_H
#define MATHUTILS_H
#include <cstdint>
#include <glm/vec3.hpp>

//  in case you dont wanna use CUDA's built-in math functions or just dont want to bother with libraries

namespace MathUtils
{
    static constexpr float pi = 3.1415926535f;
    
    float fi_sqrt( float number );   //  Quake 3's fast inverse square root
    float approx_sqrt(float number);
    uint32_t pcg_hash(uint32_t input);  //  quick random number generator
    float randomFloat(uint32_t& seed);
    void BuildOrthonormalBasis(const glm::vec3& n, glm::vec3& tangent, glm::vec3& bitangent);
    glm::vec3 CosineSampleHemisphere(const glm::vec3& normal, uint32_t& seed);
    constexpr inline float CosineHemispherePDF(float cosTheta){ return cosTheta / pi;}
    glm::vec3 UniformSampleHemisphere(const glm::vec3& normal, uint32_t& seed);
    constexpr inline float UniformHemispherePDF() {return 1 / (2 * pi);}

}


#endif
