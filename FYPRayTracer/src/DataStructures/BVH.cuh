#ifndef BVH_H
#define BVH_H

#define GLM_FORCE_CUDA
#include <vector>
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
        size_t objectIndex = -1;
        //int parentIndex;
        size_t child1 = -1;
        size_t child2 = -1;
        bool isLeaf;    //  false = internal node/sector, true = leaf node/collision object

        __host__ __device__ Node()
        : box(AABB()), objectIndex(-1), child1(-1), child2(-1), isLeaf(false) {}

        __host__ __device__ Node(const size_t objectIndex, const AABB& box)
            : box(box), objectIndex(objectIndex), child1(-1), child2(-1), isLeaf(true) {}

        __host__ __device__ Node(const size_t leftChild, const size_t rightChild, const AABB& box)
            : box(box), objectIndex(-1), child1(leftChild), child2(rightChild), isLeaf(false) {}

        __host__ __device__ Node(const size_t leftChild, const size_t rightChild)
            : box(AABB()), objectIndex(-1), child1(leftChild), child2(rightChild), isLeaf(false) {}

    };
    
    std::vector<Node> nodes;
    size_t rootIndex = -1;
    
    void TraverseRecursive(std::vector<size_t>& collisionList, const AABB& queryAABB, size_t objectQueryIndex, size_t nodeIndex) const;
    void TraverseRayRecursive(std::vector<size_t>& collisionList, const Ray& ray, size_t nodeIndex) const;
    bool IntersectRayAABB(const Ray& ray, const AABB& box) const;

    //OMP
    void OMP_ClearBVH();
    void OMP_ConstructBVHInParallel(std::vector<Node>& objects);
    size_t OMP_BuildHierarchyInParallel(std::vector<Node>& objects, size_t objectCount);
    void OMP_AssignMortonCodes(size_t objectCount);
    void OMP_BuildLeafNodes(BVH::Node* ptr_nodes, size_t objectCount);
    void OMP_BuildInternalNodes(BVH::Node* ptr_nodes, size_t objectCount);


    //  CUDA
    __host__ void CUDA_ConstructBVHInParallel(std::vector<Node>& objects);
    __host__ void CUDA_ClearBVH();
    __host__ void CUDA_AllocateMemory(size_t currentObjCount);
    __host__ void CUDA_FreeDeviceSpaceForBVH();
    __host__ size_t CUDA_BuildHierarchyInParallel(Node* objects, size_t objectCount);
    __host__ void CUDA_SortMortonCodes(size_t objectCount);
    __host__ void CUDA_CopyComponentsFromHostToDevice(BVH::Node* objects);
    __host__ void CUDA_CopyDeviceNodesToHost();

};

__host__ __device__ int findSplit(BVH::MortonCodeEntry* morton, int first, int last);
__host__ __device__ int2 determineRange(BVH::MortonCodeEntry* p_sortedMortonCodes, int objectCount, int idx);
__global__ void CUDA_AssignMortonCodesKernel(BVH::MortonCodeEntry* d_ptr_sortedMortonCodes, BVH::Node* d_ptr_collisionObjects,size_t objectCount, Vector3f minSceneBound, Vector3f maxSceneBound);
__global__ void CUDA_BuildLeafNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes, BVH::Node* ptr_nodes, AABB* ptr_objectAABBs, size_t objectCount);
__global__ void CUDA_BuildInternalNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes, BVH::Node* ptr_nodes, size_t objectCount);


#endif