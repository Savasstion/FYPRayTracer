#include "BVH.cuh"
#include <iostream>

void BVH::TraverseRecursive(std::vector<size_t>& collisionList, const AABB& queryAABB, size_t objectQueryIndex,
                            size_t nodeIndex) const
{
    if (nodeIndex >= nodes.size()) {
        std::cerr << "[ERROR] nodeIndex out of bounds: " << nodeIndex << " (size: " << nodes.size() << ")\n";
        return;
    }

    const auto& node = nodes[nodeIndex];

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
        if (node.child1 >= nodes.size() || node.child2 >= nodes.size()) {
            std::cerr << "[ERROR] Invalid child index at node " << nodeIndex
                << " | child1: " << node.child1 << ", child2: " << node.child2
                << ", s_nodes.size(): " << nodes.size() << "\n";
            return;
        }

        TraverseRecursive(collisionList, queryAABB, objectQueryIndex, node.child1);
        TraverseRecursive(collisionList, queryAABB, objectQueryIndex, node.child2);
    }
}

void BVH::TraverseRayRecursive(std::vector<size_t>& collisionList, const Ray& ray, size_t nodeIndex) const
{
    if (nodeIndex >= nodes.size())
    {
        std::cerr << "[ERROR] nodeIndex out of bounds: " << nodeIndex << " (size: " << nodes.size() << ")\n";
        return;
    }

    const Node& node = nodes[nodeIndex];

    // If ray misses this node's AABB, skip
    if (!IntersectRayAABB(ray, node.box)) return;

    if (node.isLeaf)
    {
        // if leaf node hit, add object index
        collisionList.emplace_back(node.objectIndex);
    }
    else
    {
        // Debug checks for child bounds
        if (node.child1 >= nodes.size() || node.child2 >= nodes.size()) {
            std::cerr << "[ERROR] Invalid child index at node " << nodeIndex
                << " | child1: " << node.child1 << ", child2: " << node.child2
                << ", s_nodes.size(): " << nodes.size() << "\n";
            return;
        }
        
        // Recurse both children
        TraverseRayRecursive(collisionList, ray, node.child1);
        TraverseRayRecursive(collisionList, ray, node.child2);
    }
}

bool BVH::IntersectRayAABB(const Ray& ray, const AABB& box) const
{
    float tMin = -std::numeric_limits<float>::infinity();
    float tMax = std::numeric_limits<float>::infinity();

    for (int i = 0; i < 3; ++i)
    {
        float origin = ray.origin[i];
        float direction = ray.direction[i];
        float invD = 1.0f / direction;

        float t0 = (box.lowerBound[i] - origin) * invD;
        float t1 = (box.upperBound[i] - origin) * invD;

        // Swap if direction is negative
        if (invD < 0.0f)
            std::swap(t0, t1);

        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);

        // No intersection if slab range is invalid
        if (tMax < tMin)
            return false;
    }

    return true;
}


