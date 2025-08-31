#ifndef LIGHT_TREE_H
#define LIGHT_TREE_H
#include <vector>
#include "../Classes/BaseClasses/AABB.cuh"
#include "../Classes/BaseClasses/ConeBounds.h"

class LightTree
{
public:
    struct SampledLight { uint32_t emitterIndex; float pmf; };
    struct SplitCandidate { int axis; float pos; float cost; };
    struct Node
    {
        float energy = 0.0f;
        uint32_t numEmitters = 0;
        uint32_t offset = 0;    // >= 0 left child , otherwise emmiter offset
        ConeBounds bounds_o;
        AABB bounds_w;
        glm::vec3 position{0.0f};

        bool isLeaf = false;
        uint32_t triangleIndex = static_cast<uint32_t>(-1);

        Node(uint32_t triIndex, const glm::vec3& barycentricCoord, const AABB& box, const ConeBounds& orient, float emittedEnergy)
        : energy(emittedEnergy), numEmitters(1), offset(0), bounds_o(orient), bounds_w(box), position(barycentricCoord), isLeaf(true), triangleIndex(triIndex)
        {}
        Node() = default;
        
    };

    std::vector<Node> nodes;
    uint32_t leafThreshold = 1;
    std::vector<uint32_t> emitterNodeIndices;
    
    
};

#endif