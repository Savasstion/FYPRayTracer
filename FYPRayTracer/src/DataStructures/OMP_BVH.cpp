#include <algorithm>

#include "BVH.h"
#include "../Utility/MortonCode.h"

//  OMP variable
namespace 
{
    std::vector<BVH::MortonCodeEntry>  omp_sortedMortonCodes;
    std::vector<AABB> omp_AABBs;
}

void BVH::OMP_ClearBVH()
{
    nodes.clear();
    omp_sortedMortonCodes.clear();
    omp_AABBs.clear();
    rootIndex = -1;
}

void BVH::OMP_ConstructBVHInParallel(std::vector<Node>& objects)
{
    OMP_ClearBVH();
    if (!objects.empty())
        rootIndex = OMP_BuildHierarchyInParallel(objects, objects.size());
}

void BVH::OMP_AssignMortonCodes(size_t objectCount)
{
    #pragma omp parallel for
    for(int i = 0 ; i < objectCount; i++)
    {
        omp_sortedMortonCodes[i].objectIndex = i;
    
        float x = omp_AABBs[i].centroidPos.x;
        float y = omp_AABBs[i].centroidPos.y;
        float z = omp_AABBs[i].centroidPos.z;
        omp_sortedMortonCodes[i].mortonCode = morton3D(x, y, z);
    }
    
}

void BVH::OMP_BuildLeafNodes(BVH::Node* ptr_nodes, size_t objectCount)
{
    // first N indices will be leaf nodes
    #pragma omp parallel for
    for (int idx = 0; idx < objectCount; idx++) // in parallel
    {
        //leafNodes[idx].objectID = sortedObjectIDs[idx];
        ptr_nodes[idx] = BVH::Node(omp_sortedMortonCodes[idx].objectIndex, omp_AABBs[omp_sortedMortonCodes[idx].objectIndex]);   //  Leaf node constructor
    }
}

void BVH::OMP_BuildInternalNodes(BVH::Node* ptr_nodes, size_t objectCount)
{
    #pragma omp parallel for
    for (int idx = 0; idx < objectCount - 1; idx++) // in parallel
    {
        // Find out which range of objects the node corresponds to.

        //  may need to convert idx to 32-bit int
        int2 range = determineRange(omp_sortedMortonCodes.data(), objectCount, idx);
        int first = range.x;
        int last = range.y;
        
        //  get all encompassing leaf node's AABBs within the range
        AABB internalNodeBox = ptr_nodes[range.x].box;
        for (int i = range.x + 1; i <= range.y; i++)
        {
            if(ptr_nodes[i].isLeaf)
                internalNodeBox = AABB::UnionAABB(internalNodeBox, ptr_nodes[i].box);
        }

        // Determine where to split the range.
        int split = findSplit(omp_sortedMortonCodes.data(), first, last);

        // Select childA.
        size_t indexA;
        if (split == first)
        {
            indexA = split;
        }
        else
        {
            indexA = objectCount + split;
        }
        
        // Select childB.
        size_t indexB;
        if (split + 1 == last)
        {
            indexB = split + 1;
        }
        else
        {
            indexB = objectCount + split + 1;
        }
        
        // Record parent-child relationships.
        ptr_nodes[idx + objectCount] = BVH::Node(indexA, indexB, internalNodeBox);  
    }
}

size_t BVH::OMP_BuildHierarchyInParallel(std::vector<Node>& objects, size_t objectCount)
{
    //get all AABBs of all objects, need them for leaf node initialization later
    omp_AABBs.assign(objectCount, AABB());
    #pragma omp parallel for
    for (int i = 0; i < objectCount; i++)
    {
        omp_AABBs[i] = (objects[i].box);
    }
    
    // Assign Morton Codes
    omp_sortedMortonCodes.assign(objectCount, MortonCodeEntry());
    OMP_AssignMortonCodes(objectCount);

    //  Sort Morton Codes
    std::sort(omp_sortedMortonCodes.data(), omp_sortedMortonCodes.data() + objectCount, [](const MortonCodeEntry& a, const MortonCodeEntry& b) {
        return a.mortonCode < b.mortonCode;
    });

    //  Total bvh nodes will always be 2 * N - 1
    //  leaf node count = N
    //  internal node count = N - 1 
    nodes.assign(2 * objectCount - 1, Node());
    //  first N nodes in p_ptr_nodes are leafs, the next N - 1 nodes will be internal nodes
    OMP_BuildLeafNodes(nodes.data(), objectCount);
    OMP_BuildInternalNodes(nodes.data(), objectCount);

    
    return objectCount; //root always N
}
