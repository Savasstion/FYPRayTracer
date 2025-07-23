#ifndef BVH_H
#define BVH_H

#include <vector>
#include "../Classes/BaseClasses/AABB.h"
#include <omp.h>
#include "../Classes/BaseClasses/Vector3f.h"


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
        Vector3f centroidPos{0,0,0};
        size_t objectIndex = -1;
        //int parentIndex;
        size_t child1 = -1;
        size_t child2 = -1;
        bool isLeaf;    //  false = internal node/sector, true = leaf node/collision object

        Node()
        : box(AABB()), centroidPos(Vector3f()), objectIndex(-1), child1(-1), child2(-1), isLeaf(false) {}

        Node(const size_t objectIndex, const AABB& box)
            : box(box), centroidPos(Vector3f()), objectIndex(objectIndex), child1(-1), child2(-1), isLeaf(true) {}

        Node(const size_t leftChild, const size_t rightChild, const AABB& box, const Vector3f& centroidPos)
            : box(box), centroidPos(centroidPos), objectIndex(-1), child1(leftChild), child2(rightChild), isLeaf(false) {}

        Node(const size_t leftChild, const size_t rightChild)
            : box(AABB()), centroidPos(Vector3f()), objectIndex(-1), child1(leftChild), child2(rightChild), isLeaf(false) {}

    };
    
    std::vector<Node> nodes;
    size_t rootIndex = -1;
    
    void TraverseRecursive(std::vector<size_t>& collisionList, const AABB& queryAABB, size_t objectQueryIndex, size_t nodeIndex);
    //OMP
    void OMP_ClearBVH();
    void OMP_ConstructBVHInParallel(std::vector<Node>& objects);
    size_t OMP_BuildHierarchyInParallel(std::vector<Node>& objects, size_t objectCount);
    void OMP_AssignMortonCodes(size_t objectCount);
    void OMP_BuildLeafNodes(BVH::Node* ptr_nodes, size_t objectCount);
    void OMP_BuildInternalNodes(BVH::Node* ptr_nodes, size_t objectCount);

};




#endif