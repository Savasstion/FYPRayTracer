#ifndef LIGHT_TREE_CUH
#define LIGHT_TREE_CUH
#include <cuda.h>
#define GLM_FORCE_CUDA
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
    struct Bin
    {
        AABB bounds_w;           // spatial bounds of emitters in this bin
        ConeBounds bounds_o;     // orientation cone of emitters
        float energy = 0.0f;     // total emitted energy (flux)
        uint32_t numEmitters = 0;

        void Reset()
        {
            bounds_w = AABB();
            bounds_o = ConeBounds();
            energy = 0.0f;
            numEmitters = 0;
        }

        void AddEmitter(const Node& emitter)
        {
            bounds_w = AABB::UnionAABB(bounds_w, emitter.bounds_w);
            bounds_o = ConeBounds::UnionCone(bounds_o, emitter.bounds_o);
            energy += emitter.energy;
            numEmitters++;
        }
    };

    Node* nodes = nullptr;
    uint32_t nodeCount = 0;
    uint32_t rootIndex = static_cast<uint32_t>(-1);
    

    void ConstructLightTree(Node* objects, uint32_t objCount);
    uint32_t BuildHierarchyRecursively_SAOH(Node* outNodes, uint32_t& outCount, Node* work, uint32_t first, uint32_t last);
    void ClearLightTree();
    void AllocateNodes(uint32_t count);
    void FreeNodes();
    float GetOrientationBoundsAreaMeasure(const float& theta_o, const float& theta_e);
    float GetProbabilityOfSamplingCluster(float area, float orientBoundAreaMeasure, float energy);
    float GetSplitCost(float probLeftCluster, float probRightCluster, float probCluster);
};

#endif