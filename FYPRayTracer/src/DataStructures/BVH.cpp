#include "BVH.cuh"
#include <iostream>
#include <algorithm>
#include <limits>

void BVH::TraverseRecursive(size_t*& collisionList, size_t& collisionCount,
    const AABB& queryAABB, size_t objectQueryIndex,
    size_t nodeIndex) const
{
    if (nodeIndex >= nodeCount) {
        std::cerr << "[ERROR] nodeIndex out of bounds: " << nodeIndex << " (nodeCount: " << nodeCount << ")\n";
        return;
    }

    const Node& node = nodes[nodeIndex];

    // Check for AABB overlap
    if (!AABB::isIntersect(queryAABB, node.box))
        return;

    if (node.isLeaf)
    {
        if (node.objectIndex != objectQueryIndex)
            collisionList[collisionCount++] = node.objectIndex;
    }
    else
    {
        // Bounds check for children
        if (node.child1 >= nodeCount || node.child2 >= nodeCount) {
            std::cerr << "[ERROR] Invalid child index at node " << nodeIndex
                << " | child1: " << node.child1 << ", child2: " << node.child2
                << ", nodeCount: " << nodeCount << "\n";
            return;
        }

        TraverseRecursive(collisionList, collisionCount, queryAABB, objectQueryIndex, node.child1);
        TraverseRecursive(collisionList, collisionCount, queryAABB, objectQueryIndex, node.child2);
    }
}

void BVH::TraverseRayRecursive(size_t*& collisionList, size_t& collisionCount,
    const Ray& ray, size_t nodeIndex) const
{
    if (nodeIndex >= nodeCount) {
        std::cerr << "[ERROR] nodeIndex out of bounds: " << nodeIndex << " (nodeCount: " << nodeCount << ")\n";
        return;
    }

    const Node& node = nodes[nodeIndex];

    if (!IntersectRayAABB(ray, node.box))
        return;

    if (node.isLeaf)
    {
        collisionList[collisionCount++] = node.objectIndex;
    }
    else
    {
        if (node.child1 >= nodeCount || node.child2 >= nodeCount) {
            std::cerr << "[ERROR] Invalid child index at node " << nodeIndex
                << " | child1: " << node.child1 << ", child2: " << node.child2
                << ", nodeCount: " << nodeCount << "\n";
            return;
        }

        TraverseRayRecursive(collisionList, collisionCount, ray, node.child1);
        TraverseRayRecursive(collisionList, collisionCount, ray, node.child2);
    }
}

void BVH::ConstructBVH(Node* objects, size_t objCount)
{
    FreeHostNodes();
    objectCount = objCount;

    if (objectCount > 0)
    {
        // Allocate host nodes: total 2 * N - 1
        AllocateHostNodes(2 * objectCount - 1);

        Node* work = new Node[objectCount];
        for (size_t i = 0; i < objectCount; ++i) {
            work[i] = Node(objects[i].objectIndex, objects[i].box);
        }

        size_t outCount = 0;

        rootIndex = BuildHierarchyRecursively(nodes, outCount, work, 0, objectCount);

        delete[] work;
        nodeCount = outCount;

    }
}

void BVH::ClearBVH()
{
    FreeHostNodes();  // free the allocated nodes array
    rootIndex = static_cast<size_t>(-1);
}

int BVH::LargestExtentAxis(const AABB& b)
{
    const float ex = b.upperBound.x - b.lowerBound.x;
    const float ey = b.upperBound.y - b.lowerBound.y;
    const float ez = b.upperBound.z - b.lowerBound.z;

    if (ex > ey && ex > ez) return 0; // x
    if (ey > ez)             return 1; // y
    return 2;                          // z
}

AABB BVH::RangeBounds(BVH::Node* arr, size_t first, size_t last)
{
    AABB out = arr[first].box;
    for (size_t i = first + 1; i < last; ++i) {
        out = AABB::UnionAABB(out, arr[i].box);
    }
    return out;
}

size_t BVH::BuildHierarchyRecursively(BVH::Node* outNodes, size_t& outCount, BVH::Node* work, size_t first, size_t last)
{
    const size_t count = last - first;

    // leaf node
    if (count == 1) {
        outNodes[outCount] = BVH::Node(work[first].objectIndex, work[first].box);
        return outCount++;
    }

    // bounds + split axis
    AABB bounds = RangeBounds(work, first, last);
    const int axis = LargestExtentAxis(bounds);

    // median split by centroid
    const size_t mid = (first + last) / 2;
    std::nth_element(work + first, work + mid, work + last,
                     [axis](const BVH::Node& a, const BVH::Node& b) {
                         auto aabbA = a.box;
                         auto aabbB = b.box;
                         return AABB::FindCentroid(aabbA)[axis] < AABB::FindCentroid(aabbB)[axis];
                     });

    // build children
    const size_t leftIndex  = BuildHierarchyRecursively(outNodes, outCount, work, first, mid);
    const size_t rightIndex = BuildHierarchyRecursively(outNodes, outCount, work, mid,   last);

    // parent
    const AABB parentBox = AABB::UnionAABB(outNodes[leftIndex].box, outNodes[rightIndex].box);
    outNodes[outCount] = BVH::Node(leftIndex, rightIndex, parentBox);
    return outCount++;
}
