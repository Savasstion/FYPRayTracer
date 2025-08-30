#ifndef LIGHT_TREE_H
#define LIGHT_TREE_H
#include <cstdint>
#include <glm/vec3.hpp>

class LightTree
{
public:
    
    struct Node
    {
        struct OrientationBounds
        {
            glm::vec3 axis{0.0f};
            float theta_o = 0.0f;
            float theta_e = 0.0f;
        };

        struct WorldSpaceBounds
        {
            glm::vec3 upperBounds{0.0f};
            glm::vec3 lowerBounds{0.0f};
        };
        
        float energy = 0.0f;
        uint32_t numEmitters = 0;
        uint32_t offset = 0;    // >= 0 left child , otherwise emmiter offset
        OrientationBounds bounds_o;
        WorldSpaceBounds bounds_w;
    };
    
};

#endif