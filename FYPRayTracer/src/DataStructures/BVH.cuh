#ifndef BVH_H
#define BVH_H
#include <cuda.h>
#define GLM_FORCE_CUDA
#include "../Classes/BaseClasses/AABB.cuh"
#include <omp.h>
#include "../Classes/BaseClasses/Vector3f.cuh"
#include "../Classes/BaseClasses/Ray.h"

class BVH
{
public:
    struct MortonCodeEntry
    {
        unsigned int mortonCode;
        size_t objectIndex;
    };

    struct Node
    {
        AABB box;
        size_t objectIndex = static_cast<size_t>(-1);
        size_t child1 = static_cast<size_t>(-1);
        size_t child2 = static_cast<size_t>(-1);
        bool isLeaf;

        __host__ __device__ Node()
            : box(AABB()), objectIndex(static_cast<size_t>(-1)),
              child1(static_cast<size_t>(-1)), child2(static_cast<size_t>(-1)),
              isLeaf(false) {}

        __host__ __device__ Node(const size_t objectIndex, const AABB& box)
            : box(box), objectIndex(objectIndex),
              child1(static_cast<size_t>(-1)), child2(static_cast<size_t>(-1)),
              isLeaf(true) {}

        __host__ __device__ Node(const size_t leftChild, const size_t rightChild, const AABB& box)
            : box(box), objectIndex(static_cast<size_t>(-1)),
              child1(leftChild), child2(rightChild), isLeaf(false) {}

        __host__ __device__ Node(const size_t leftChild, const size_t rightChild)
            : box(AABB()), objectIndex(static_cast<size_t>(-1)),
              child1(leftChild), child2(rightChild), isLeaf(false) {}
    };


    Node* nodes = nullptr;
    size_t nodeCount = 0;
    bool isNewValuesSet = true;
    size_t objectCount = 0;
    size_t rootIndex = static_cast<size_t>(-1);

    // buffers for CUDA parallelization, not needed to copy 
    BVH::Node* d_ptr_nodes = nullptr;
    BVH::Node* d_ptr_collisionObjects = nullptr;
    BVH::MortonCodeEntry* h_ptr_sortedMortonCodes = nullptr;
    BVH::MortonCodeEntry* d_ptr_sortedMortonCodes = nullptr;
    AABB* d_ptr_objectAABBs = nullptr;


    // Traversal
    void TraverseRecursive(size_t*& collisionList, size_t& collisionCount,
                            const AABB& queryAABB, size_t objectQueryIndex, size_t nodeIndex) const;
    void TraverseRayRecursive(size_t*& collisionList, size_t& collisionCount,
                               const Ray& ray, size_t nodeIndex) const;
    bool IntersectRayAABB(const Ray& ray, const AABB& box) const;

    // OMP
    void OMP_ClearBVH();
    void OMP_ConstructBVHInParallel(Node* objects, size_t objectCount);
    size_t OMP_BuildHierarchyInParallel(Node* objects, size_t objectCount);
    void OMP_AssignMortonCodes(size_t objectCount);
    void OMP_BuildLeafNodes(BVH::Node* ptr_nodes, size_t objectCount);
    void OMP_BuildInternalNodes(BVH::Node* ptr_nodes, size_t objectCount);

    // CUDA
    __host__ void CUDA_ConstructBVHInParallel(Node* objects, size_t objectCount);
    __host__ void CUDA_ClearBVH();
    __host__ void CUDA_AllocateMemory(size_t currentObjCount);
    __host__ void CUDA_FreeDeviceSpaceForBVH();
    __host__ size_t CUDA_BuildHierarchyInParallel(Node* objects, size_t objectCount);
    __host__ void CUDA_SortMortonCodes(size_t objectCount);
    __host__ void CUDA_CopyComponentsFromHostToDevice(BVH::Node* objects);
    __host__ void CUDA_CopyDeviceNodesToHost();

    // Utility to allocate host-side nodes
    void AllocateHostNodes(size_t count) {
        if (nodes) delete[] nodes;
        nodes = new Node[count];
        nodeCount = count;
    }

    // Utility to free host-side nodes
    void FreeHostNodes() {
        if (nodes) {
            delete[] nodes;
            nodes = nullptr;
            nodeCount = 0;
        }
    }
    
};

__host__ __device__ int findSplit(BVH::MortonCodeEntry* morton, int first, int last);
__host__ __device__ int2 determineRange(BVH::MortonCodeEntry* p_sortedMortonCodes, int objectCount, int idx);

__global__ void CUDA_AssignMortonCodesKernel(BVH::MortonCodeEntry* d_ptr_sortedMortonCodes,
                                             BVH::Node* d_ptr_collisionObjects,
                                             size_t objectCount,
                                             Vector3f minSceneBound, Vector3f maxSceneBound);

__global__ void CUDA_BuildLeafNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes,
                                          BVH::Node* ptr_nodes,
                                          AABB* ptr_objectAABBs,
                                          size_t objectCount);

__global__ void CUDA_BuildInternalNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes,
                                              BVH::Node* ptr_nodes,
                                              size_t objectCount);

__host__ __forceinline__ BVH* BVHToGPU(BVH h_bvh)
{
    BVH* d_bvh = nullptr;

    // Allocate BVH struct on device
    cudaMalloc(&d_bvh, sizeof(BVH));

    // Temporary copy on host to fix device pointers
    BVH temp;

    // Copy only the selected fields
    temp.nodeCount       = h_bvh.nodeCount;
    temp.isNewValuesSet  = h_bvh.isNewValuesSet;
    temp.objectCount     = h_bvh.objectCount;
    temp.rootIndex       = h_bvh.rootIndex;

    // Copy nodes array to device if available
    if (h_bvh.nodes && h_bvh.nodeCount > 0) {
        BVH::Node* d_nodes = nullptr;
        cudaMalloc(&d_nodes, h_bvh.nodeCount * sizeof(BVH::Node));
        cudaMemcpy(d_nodes, h_bvh.nodes, h_bvh.nodeCount * sizeof(BVH::Node), cudaMemcpyHostToDevice);
        temp.nodes = d_nodes;
    } else {
        temp.nodes = nullptr;
    }

    // Copy patched struct to device
    cudaMemcpy(d_bvh, &temp, sizeof(BVH), cudaMemcpyHostToDevice);

    return d_bvh;
}

__host__ __forceinline__ void FreeBVH_GPU(BVH* d_bvh)
{
    if (!d_bvh) return;

    // Copy BVH struct from device to host so we can access its pointers
    BVH h_bvh;
    cudaMemcpy(&h_bvh, d_bvh, sizeof(BVH), cudaMemcpyDeviceToHost);

    // Free all pointers
    if (h_bvh.nodes)
        cudaFree(h_bvh.nodes);
    if (h_bvh.d_ptr_nodes)
        cudaFree(h_bvh.d_ptr_nodes);
    if (h_bvh.d_ptr_collisionObjects)
        cudaFree(h_bvh.d_ptr_collisionObjects);
    if (h_bvh.h_ptr_sortedMortonCodes)
        cudaFree(h_bvh.h_ptr_sortedMortonCodes);
    if (h_bvh.d_ptr_sortedMortonCodes)
        cudaFree(h_bvh.d_ptr_sortedMortonCodes);
    if (h_bvh.d_ptr_objectAABBs)
        cudaFree(h_bvh.d_ptr_objectAABBs);

    // Free BVH struct itself
    cudaFree(d_bvh);
}


#endif
