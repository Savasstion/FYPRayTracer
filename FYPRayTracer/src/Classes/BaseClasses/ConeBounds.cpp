#include "ConeBounds.h"
#include <glm/gtx/quaternion.hpp>
#include "../../Utility/MathUtils.cuh"

ConeBounds ConeBounds::UnionCone(ConeBounds a, ConeBounds b)
{
    //  from the paper Importance Sampling of Many Lights with Adaptive Tree Splitting
    
    if (b.theta_o > a.theta_o)
    {
        //  Swap(a,b)
        ConeBounds tmp = a;
        a = b;
        b = tmp;
    }

    float theta_d = glm::acos(glm::dot(a.axis, b.axis));
    float theta_e = fmaxf(a.theta_e, b.theta_e);

    if(fminf(theta_d + b.theta_o, MathUtils::pi) <= a.theta_o)
        return {a.axis, a.theta_o, theta_e};
    
    float theta_o = (a.theta_o + theta_d + b.theta_o) * 0.5f;

    // full hemisphere?
    if(MathUtils::pi <= theta_o)
        return {a.axis, MathUtils::pi, theta_e};
        
    float theta_r = theta_o - a.theta_o;
    glm::vec3 rotAxis = glm::cross(a.axis, b.axis);
        
    glm::mat4 R = glm::rotate(glm::mat4(1.0f), theta_r, rotAxis);
    glm::vec3 axis = glm::normalize(glm::vec3(R * glm::vec4(a.axis, 0.0f)));

    return {axis, theta_o, theta_e};
    
}
