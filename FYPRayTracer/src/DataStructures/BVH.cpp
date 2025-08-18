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

