#include "BVH.h"
#include <iostream>
#include "../Utility/BitManipulation.h"

void BVH::TraverseRecursive(std::vector<size_t>& collisionList, const AABB& queryAABB, size_t objectQueryIndex,
                            size_t nodeIndex)
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

int BVH::findSplit(BVH::MortonCodeEntry* morton, int first, int last)
{
    // Identical Morton codes => split the range in the middle.

    unsigned int firstCode = morton[first].mortonCode;
    unsigned int lastCode = morton[last].mortonCode;

    if (firstCode == lastCode)
        return (first + last) >> 1;

    // Calculate the number of highest bits that are the same
    // for all objects, using the count-leading-zeros intrinsic.

    int delta_node = clz(firstCode ^ lastCode);

    // Use binary search to find where the next bit differs.
    // Specifically, we are looking for the highest object that
    // shares more than commonPrefix bits with the first one.

    int split = first; // initial guess
    int stride = last - first;

    do
    {
        stride = (stride + 1) >> 1; // exponential decrease
        int middle = split + stride; // proposed new position

        if (middle < last)
        {
            int delta = clz(firstCode ^ morton[middle].mortonCode);
            if (delta > delta_node)
                split = middle; // accept proposal
        }
    } while (stride > 1);

    return split;
}

BVH::int2 BVH::determineRange(BVH::MortonCodeEntry* p_sortedMortonCodes, int objectCount, int idx)
{
    if (idx == 0)
    {
        int2 range = { 0, objectCount - 1 };
        return range;
    }

    // Determine direction of the range
    unsigned int selfMortonCode = p_sortedMortonCodes[idx].mortonCode;
    int deltaL = clz(selfMortonCode ^ p_sortedMortonCodes[idx - 1].mortonCode);
    int deltaR = clz(selfMortonCode ^ p_sortedMortonCodes[idx + 1].mortonCode);
    int direction = (deltaR > deltaL) ? 1 : -1;


    // Compute upper bound for the length of the range
    int deltaMin = (deltaL < deltaR) ? deltaL : deltaR;
    int lmax = 2;
    int delta = -1;
    int i_tmp = idx + direction * lmax;

    if (0 <= i_tmp && i_tmp < objectCount)
    {
        delta = clz(selfMortonCode ^ p_sortedMortonCodes[i_tmp].mortonCode);
    }

    while (delta > deltaMin)
    {
        lmax <<= 1;
        i_tmp = idx + direction * lmax;
        delta = 1;
        if (0 <= i_tmp && i_tmp < objectCount)
        {
            delta = clz(selfMortonCode ^ p_sortedMortonCodes[i_tmp].mortonCode);
        }
    }

    // Find the other end using binary search
    int l = 0;
    int t = lmax >> 1;
    while (t > 0)
    {
        i_tmp = idx + (l + t) * direction;
        delta = -1;
        if (0 <= i_tmp && i_tmp < objectCount)
        {
            delta = clz(selfMortonCode ^ p_sortedMortonCodes[i_tmp].mortonCode);
        }
        if (delta > deltaMin)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * direction;
    if (direction < 0)
    {
        unsigned int tmp = idx;
        idx = jdx;
        jdx = tmp;
    }

    int2 result;
    result.x = idx;
    result.y = jdx;
    return result;
}
