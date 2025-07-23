#include "BVH.h"
#include <iostream>

void BVH::TraverseRecursive(std::vector<size_t>& collisionList, const AABB& queryAABB, size_t objectQueryIndex,
                            size_t nodeIndex)
{
    if (nodeIndex >= s_nodes.size()) {
        std::cerr << "[ERROR] nodeIndex out of bounds: " << nodeIndex << " (size: " << s_nodes.size() << ")\n";
        return;
    }

    const auto& node = s_nodes[nodeIndex];

    // Check if overlap
    if (!AABB::isIntersect(queryAABB, node.box))
        return;

    if (node.isLeaf)
    {
        if (node.objectIndex == objectQueryIndex)
            return;

        collisionList.emplace_back(node.objectIndex);
    }
    else
    {
        // Debug checks for child bounds
        if (node.child1 >= s_nodes.size() || node.child2 >= s_nodes.size()) {
            std::cerr << "[ERROR] Invalid child index at node " << nodeIndex
                << " | child1: " << node.child1 << ", child2: " << node.child2
                << ", s_nodes.size(): " << s_nodes.size() << "\n";
            return;
        }

        TraverseRecursive(collisionList, queryAABB, objectQueryIndex, node.child1);
        TraverseRecursive(collisionList, queryAABB, objectQueryIndex, node.child2);
    }
}
