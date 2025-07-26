#ifndef BVH_H
#define BVH_H

#include <vector>
#include "../Classes/BaseClasses/AABB.h"
#include <omp.h>
#include "../Classes/BaseClasses/Vector3f.h"
#include "../Classes/BaseClasses/Ray.h"

class BVH
{
public:
    struct int2
    {
        int x, y;
    };
    
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
    int findSplit(BVH::MortonCodeEntry* morton, int first, int last);
    int2 determineRange(BVH::MortonCodeEntry* p_sortedMortonCodes, int objectCount, int idx);

    //OMP
    void OMP_ClearBVH();
    void OMP_ConstructBVHInParallel(std::vector<Node>& objects);
    size_t OMP_BuildHierarchyInParallel(std::vector<Node>& objects, size_t objectCount);
    void OMP_AssignMortonCodes(size_t objectCount);
    void OMP_BuildLeafNodes(BVH::Node* ptr_nodes, size_t objectCount);
    void OMP_BuildInternalNodes(BVH::Node* ptr_nodes, size_t objectCount);

    //  CUDA
    //__global__ void BuildLeafNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes, BVH::Node* ptr_nodes, AABB* ptr_objectAABBs, size_t objectCount);
    //__global__ void BuildInternalNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes, BVH::Node* ptr_nodes, size_t objectCount);
    //__global__ void AssignMortonCodesKernel(BVH::MortonCodeEntry* d_ptr_sortedMortonCodes, CircleColliderComponent* d_ptr_circle_collider_components, Entity* d_ptr_entities, TransformComponent* d_ptr_transform_components, size_t objectCount);
    //__global__ void SortMortonCodesKernel(BVH::MortonCodeEntry* data, int n);
    //__host__ void AssignInternalNodeAABB(size_t nodeIndex, BVH::Node* nodes, size_t nodeCount);
    //__host__ void AssignInternalNodeAABB_Updated(BVH::Node* nodes, size_t objectCount);
    //__host__ __device__ int findSplit(BVH::MortonCodeEntry* morton, int first, int last);
    //__host__ __device__ int2 determineRange(BVH::MortonCodeEntry* p_sortedMortonCodes, int objectCount, int idx);

};




#endif