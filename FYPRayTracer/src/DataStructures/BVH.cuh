#ifndef BVH_H
#define BVH_H
#include <cuda.h>
#define GLM_FORCE_CUDA
#include <iostream>

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
    size_t objectOffset = 0;    //  when building BLASes, nodes may not start from 0 first
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

__host__ __device__ __forceinline__ bool IntersectRayAABB(const Ray& ray, const AABB& box)
{
    float tMin = 0.0f;
    float tMax = FLT_MAX;

    for (int i = 0; i < 3; ++i)
    {
        float origin = ray.origin[i];
        float direction = ray.direction[i];

        if (direction == 0.0f)
        {
            // Ray is parallel to this slab; if origin not within bounds, no intersection
            if (origin < box.lowerBound[i] || origin > box.upperBound[i])
                return false;
            
            // Parallel and inside slab; t0/t1 are unbounded
            continue;
        }

        float invD = 1.0f / direction;
        float t0 = (box.lowerBound[i] - origin) * invD;
        float t1 = (box.upperBound[i] - origin) * invD;

        if (invD < 0.0f)
        {
            //swap(t0, t1);
            auto tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
            

        tMin = fmaxf(tMin, t0);
        tMax = fminf(tMax, t1);

        if (tMax < tMin)
            return false;
    }

    return true;
}
__host__ __device__ int findSplit(BVH::MortonCodeEntry* morton, int first, int last);
__host__ __device__ int2 determineRange(BVH::MortonCodeEntry* p_sortedMortonCodes, int objectCount, int idx);

__global__ void CUDA_AssignMortonCodesKernel(BVH::MortonCodeEntry* d_ptr_sortedMortonCodes,
                                             BVH::Node* d_ptr_collisionObjects,
                                             size_t objectCount,
                                             Vector3f minSceneBound, Vector3f maxSceneBound);

__global__ void CUDA_BuildLeafNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes,
                                          BVH::Node* ptr_nodes,
                                          AABB* ptr_objectAABBs,
                                          size_t objectCount,
                                          size_t objectOffset);

__global__ void CUDA_BuildInternalNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes,
                                              BVH::Node* ptr_nodes,
                                              size_t objectCount);

__host__ __forceinline__ BVH* BVHToGPU(const BVH& h_bvh)
{
    cudaError_t err;

    // Allocate BVH struct on device
    BVH* d_bvh = nullptr;
    err = cudaMalloc(&d_bvh, sizeof(BVH));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc BVH error: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    // Prepare temporary host copy, zero-initialized
    BVH temp{};
    
    // Copy only the necessary fields
    temp.nodeCount      = h_bvh.nodeCount;
    temp.isNewValuesSet = h_bvh.isNewValuesSet;
    temp.objectCount    = h_bvh.objectCount;

    // Safe rootIndex assignment
    if (h_bvh.nodeCount > 0) {
        // Use CPU rootIndex if valid, otherwise default to 0
        temp.rootIndex = (h_bvh.rootIndex == static_cast<size_t>(-1)) ? 0 : h_bvh.rootIndex;
    } else {
        temp.rootIndex = static_cast<size_t>(-1);
    }

    // Copy nodes array to device if available
    if (h_bvh.nodes && h_bvh.nodeCount > 0) {
        BVH::Node* d_nodes = nullptr;
        err = cudaMalloc(&d_nodes, h_bvh.nodeCount * sizeof(BVH::Node));
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc BVH nodes error: " << cudaGetErrorString(err) << std::endl;
        } else {
            err = cudaMemcpy(d_nodes, h_bvh.nodes,
                             h_bvh.nodeCount * sizeof(BVH::Node),
                             cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                std::cerr << "cudaMemcpy BVH nodes error: " << cudaGetErrorString(err) << std::endl;
            }
        }
        temp.nodes = d_nodes;
    } else {
        temp.nodes = nullptr;
    }

    // Null out all unused GPU/host buffers
    temp.d_ptr_nodes             = nullptr;
    temp.d_ptr_collisionObjects  = nullptr;
    temp.h_ptr_sortedMortonCodes = nullptr;
    temp.d_ptr_sortedMortonCodes = nullptr;
    temp.d_ptr_objectAABBs       = nullptr;

    // Copy fully patched struct to device
    err = cudaMemcpy(d_bvh, &temp, sizeof(BVH), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy BVH struct error: " << cudaGetErrorString(err) << std::endl;
    }

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
