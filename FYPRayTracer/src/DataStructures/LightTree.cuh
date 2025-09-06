#ifndef LIGHT_TREE_CUH
#define LIGHT_TREE_CUH
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <iostream>
#include <vector>
#include "../Classes/BaseClasses/AABB.cuh"
#include "../Classes/BaseClasses/ConeBounds.cuh"

//  forward declare
struct Mesh_GPU;

class LightTree
{
public:
    struct ShadingPointQuery{ glm::vec3 position; glm::vec3 normal; };
    struct SampledLight { uint32_t emitterIndex; float pmf; };
    struct Node
    {
        float energy = 0.0f;
        uint32_t numEmitters = 0;
        uint32_t offset = 0;    // >= 0 left child , otherwise emmiter offset
        ConeBounds bounds_o;
        AABB bounds_w;
        glm::vec3 position{0.0f};

        bool isLeaf = false;
        uint32_t emmiterIndex = static_cast<uint32_t>(-1);  //  if leaf, this stores the index of actual emmisive triangle. If internal node, then this stores the right child index

        Node(uint32_t triIndex, const glm::vec3& barycentricCoord, const AABB& box, const ConeBounds& orient, float emittedEnergy)
        : energy(emittedEnergy), numEmitters(1), offset(0), bounds_o(orient), bounds_w(box), position(barycentricCoord), isLeaf(true), emmiterIndex(triIndex)
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
__host__ __device__ __forceinline__ float ComputeClusterImportance(const LightTree::ShadingPointQuery& shadingPoint, const LightTree::Node& cluster)
{
    //  refer to the paper, Importance Sampling of Many Lights with Adaptive Tree Splitting at 5.1 Cluster Importance for Surfaces
    float theta_u;
     {
         ConeBounds cone = ConeBounds::FindConeThatEnvelopsAABBFromPoint(cluster.bounds_w, shadingPoint.position);
         theta_u = cone.theta_o;
     }

    glm::vec3 aabbPos;
    aabbPos.x = cluster.bounds_w.centroidPos.x;
    aabbPos.y = cluster.bounds_w.centroidPos.y;
    aabbPos.z = cluster.bounds_w.centroidPos.z;
    glm::vec3 dir = shadingPoint.position- aabbPos;
    float distanceSquared = fmaxf(glm::dot(dir, dir), 1e-12f);  //1e-12f is there to avoid zero division
    dir = glm::normalize(dir);
    
    float dotVal = glm::dot(cluster.bounds_o.axis, dir);
    dotVal = glm::clamp(dotVal, -1.0f, 1.0f);
    float theta = acosf(dotVal);

    float angleTerm = glm::clamp(theta - cluster.bounds_o.theta_o - theta_u, 0.0f, cluster.bounds_o.theta_e);
    float cosTerm = cosf(angleTerm);

    return (cluster.energy * cosTerm) / distanceSquared;
}

__host__ __forceinline__ LightTree* LightTreeToGPU(const LightTree& h_lightTree)
{
    cudaError_t err;

    // Allocate Light Tree struct on device
    LightTree* d_lightTree = nullptr;
    err = cudaMalloc(&d_lightTree, sizeof(LightTree));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc BVH error: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    // Prepare temporary host copy, zero-initialized
    LightTree temp{};
    
    // Copy only the necessary fields
    temp.nodeCount      = h_lightTree.nodeCount;

    // Safe rootIndex assignment
    if (h_lightTree.nodeCount > 0) {
        // Use CPU rootIndex if valid, otherwise default to 0
        temp.rootIndex = (h_lightTree.rootIndex == static_cast<uint32_t>(-1)) ? 0 : h_lightTree.rootIndex;
    } else {
        temp.rootIndex = static_cast<uint32_t>(-1);
    }

    // Copy nodes array to device if available
    if (h_lightTree.nodes && h_lightTree.nodeCount > 0) {
        LightTree::Node* d_nodes = nullptr;
        err = cudaMalloc(&d_nodes, h_lightTree.nodeCount * sizeof(LightTree::Node));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc BVH nodes error: " << cudaGetErrorString(err) << std::endl;
        } else {
            err = cudaMemcpy(d_nodes, h_lightTree.nodes,
                             h_lightTree.nodeCount * sizeof(LightTree::Node),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "cudaMemcpy BVH nodes error: " << cudaGetErrorString(err) << std::endl;
            }
        }
        temp.nodes = d_nodes;
    } else {
        temp.nodes = nullptr;
    }
    
    // Copy fully patched struct to device
    err = cudaMemcpy(d_lightTree, &temp, sizeof(LightTree), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy BVH struct error: " << cudaGetErrorString(err) << std::endl;
    }

    return d_lightTree;
}

__host__ __forceinline__ void FreeLightTree_GPU(LightTree* d_lightTree)
{
    if (!d_lightTree) return;

    // Copy BVH struct from device to host so we can access its pointers
    LightTree h_lightTree;
    cudaMemcpy(&h_lightTree, d_lightTree, sizeof(LightTree), cudaMemcpyDeviceToHost);

    // Free all pointers
    if (h_lightTree.nodes)
        cudaFree(h_lightTree.nodes);

    // Free BVH struct itself
    cudaFree(d_lightTree);
}

__host__ __device__ LightTree::SampledLight PickLight_BLAS(const LightTree* blas_tree, const LightTree::ShadingPointQuery& sp, uint32_t& randSeed, LightTree::SampledLight currentSampledLight);

__host__ __device__ LightTree::SampledLight PickLight_TLAS(const Mesh_GPU* meshes, const LightTree* tlas_tree, const LightTree::ShadingPointQuery& sp, uint32_t& randSeed);

#endif
