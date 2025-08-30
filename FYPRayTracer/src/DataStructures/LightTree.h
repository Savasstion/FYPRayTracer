#ifndef LIGHT_TREE_H
#define LIGHT_TREE_H
#include "../Classes/BaseClasses/AABB.cuh"

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
        
        float energy = 0.0f;
        uint32_t numEmitters = 0;
        uint32_t offset = 0;    // >= 0 left child , otherwise emmiter offset
        OrientationBounds bounds_o;
        AABB bounds_w;
    };
    
};

#endif