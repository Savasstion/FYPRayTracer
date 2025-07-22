#ifndef BVH_H
#define BVH_H

#include <vector>
#include "../Classes/BaseClasses/AABB.h"
#include <omp.h>


//class BVH
//{
//public:
//    struct MortonCodeEntry
//    {
//        unsigned int mortonCode;
//        size_t objectIndex;
//    };
//
//    struct Node
//    {
//        AABB box;
//        size_t objectIndex = -1;
//        //int parentIndex;
//        size_t child1 = -1;
//        size_t child2 = -1;
//        bool isLeaf;    //  false = internal node/sector, true = leaf node/collision object
//
//        Node()
//            : box(AABB()), objectIndex(-1), child1(-1), child2(-1), isLeaf(false) {}
//        Node(const size_t objectIndex, const AABB& box)
//            : box(box), objectIndex(objectIndex), child1(-1), child2(-1), isLeaf(true) {}
//
//        Node(const size_t leftChild, const size_t rightChild, const AABB& box)
//            : box(box), objectIndex(-1), child1(leftChild), child2(rightChild), isLeaf(false) {}
//        Node(const size_t leftChild, const size_t rightChild)
//            : objectIndex(-1), child1(leftChild), child2(rightChild), isLeaf(false), box(AABB()) {}
//
//    };
//
//    //  Serial members
//    std::vector<Node> s_nodes;
//
//    size_t rootIndex = -1;
//
//    const std::vector<Node>& GetNodes() const { return nodes; }
//    std::vector<Node>& GetNodes() { return nodes; }
//
//
//    size_t BuildHierarchy(std::vector<Node>& objects, size_t start, size_t end);
//    size_t BuildHierarchyInParallel(Node* objects, TransformComponent* ptr_transform_components, Entity* ptr_entities, CircleColliderComponent* ptr_circle_collider_components, size_t objectCount);
//
//    void ConstructBVH(std::vector<Node>& objects);
//    void ConstructBVHInParallel(std::vector<Node>& objects, std::vector<TransformComponent>& transform_components, std::vector<Entity>& entities,
//        std::vector<CircleColliderComponent>& circle_collider_components);
//    void ClearBVH();
//    void TraverseRecursive(std::vector<size_t>& collisionList, const AABB& queryAABB, size_t objectQueryIndex, size_t nodeIndex);
//    Node* GetDeviceNodePointer();
//    Node* GetHostNodePointer();
//    void CUDA_SortMortonCodes(size_t objectCount);
//
//    //OMP
//    void OMP_ClearBVH();
//    void OMP_ConstructBVHInParallel(std::vector<Node>& objects, std::vector<TransformComponent>& transform_components, std::vector<Entity>& entities,
//        std::vector<CircleColliderComponent>& circle_collider_components);
//    size_t OMP_BuildHierarchyInParallel(std::vector<Node>& objects, std::vector<TransformComponent>& transform_components, std::vector<Entity>& entities,
//        std::vector<CircleColliderComponent>& circle_collider_components, size_t objectCount);
//
//private:
//    std::vector<Node> nodes;
//};
////  OMP
//void OMP_AssignMortonCodes(TransformComponent* ptr_transform_components, Entity* ptr_entities,
//    CircleColliderComponent* ptr_circle_collider_components, size_t objectCount);
//void OMP_BuildLeafNodes(BVH::Node* ptr_nodes, size_t objectCount);
//void OMP_BuildInternalNodes(BVH::Node* ptr_nodes, size_t objectCount);


#endif