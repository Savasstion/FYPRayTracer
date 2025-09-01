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

void BVH::ConstructBVH_MedianSplit(Node* objects, size_t objCount)
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

        rootIndex = BuildHierarchyRecursively_MedianSplit(nodes, outCount, work, 0, objectCount);

        delete[] work;
        nodeCount = outCount;

    }
}

void BVH::ConstructBVH_SAH(Node* objects, size_t objCount)
{
    FreeHostNodes();
    objectCount = objCount;

    if (objectCount > 0)
    {
        // Allocate host nodes: total 2 * N - 1
        AllocateHostNodes(2 * objectCount - 1);
        

        size_t outCount = 0;

        rootIndex = BuildHierarchyRecursively_SAH(nodes, outCount, objects, 0, objectCount);
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

size_t BVH::BuildHierarchyRecursively_MedianSplit(BVH::Node* outNodes, size_t& outCount, BVH::Node* work, size_t first, size_t last)
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
    const size_t leftIndex  = BuildHierarchyRecursively_MedianSplit(outNodes, outCount, work, first, mid);
    const size_t rightIndex = BuildHierarchyRecursively_MedianSplit(outNodes, outCount, work, mid,   last);

    // parent
    const AABB parentBox = AABB::UnionAABB(outNodes[leftIndex].box, outNodes[rightIndex].box);
    outNodes[outCount] = BVH::Node(leftIndex, rightIndex, parentBox);
    return outCount++;
}

size_t BVH::BuildHierarchyRecursively_SAH(BVH::Node* outNodes, size_t& outCount, BVH::Node* work, size_t first,
    size_t last)
{
    const size_t count = last - first;

    // Leaf node
    if (count == 1) {
        outNodes[outCount] = BVH::Node(work[first].objectIndex, work[first].box);
        return outCount++;
    }

    // Compute bounds for current range
    AABB bounds = RangeBounds(work, first, last);

    // SAH parameters
    const int numBins = 16;   // usually 8–32 bins give good results
    float bestCost = FLT_MAX;
    int bestAxis = -1;
    int bestSplitBin = -1;

    // Try splitting along each axis
    for (int axis = 0; axis < 3; axis++) {
        // Find centroid bounds along this axis
        AABB centroidBounds;
        centroidBounds.lowerBound = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
        centroidBounds.upperBound = Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (size_t i = first; i < last; i++) {
            Vector3f c = AABB::FindCentroid(work[i].box);
            centroidBounds.lowerBound[axis] = MathUtils::minFloat(centroidBounds.lowerBound[axis], c[axis]);
            centroidBounds.upperBound[axis] = MathUtils::maxFloat(centroidBounds.upperBound[axis], c[axis]);
        }

        float cmin = centroidBounds.lowerBound[axis];
        float cmax = centroidBounds.upperBound[axis];
        if (cmin == cmax) continue; // Degenerate axis → skip

        // Define bins
        Bin bins[numBins];

        // Fill bins
        for (size_t i = first; i < last; i++) {
            Vector3f c = AABB::FindCentroid(work[i].box);
            int binIdx = (int)(((c[axis] - cmin) / (cmax - cmin)) * (numBins - 1));
            bins[binIdx].count++;
            bins[binIdx].box = AABB::UnionAABB(bins[binIdx].box, work[i].box);
        }

        // Prefix sums (left to right)
        AABB leftBoxes[numBins - 1];
        size_t leftCounts[numBins - 1];
        {
            AABB cur;
            cur.lowerBound = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
            cur.upperBound = Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            size_t cnt = 0;
            for (int i = 0; i < numBins - 1; i++) {
                cur = AABB::UnionAABB(cur, bins[i].box);
                cnt += bins[i].count;
                leftBoxes[i] = cur;
                leftCounts[i] = cnt;
            }
        }

        // Suffix sums (right to left)
        AABB rightBoxes[numBins - 1];
        size_t rightCounts[numBins - 1];
        {
            AABB cur;
            cur.lowerBound = Vector3f(FLT_MAX, FLT_MAX, FLT_MAX);
            cur.upperBound = Vector3f(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            size_t cnt = 0;
            for (int i = numBins - 1; i > 0; i--) {
                cur = AABB::UnionAABB(cur, bins[i].box);
                cnt += bins[i].count;
                rightBoxes[i - 1] = cur;
                rightCounts[i - 1] = cnt;
            }
        }

        // Evaluate costs
        float parentArea = bounds.GetSurfaceArea();
        for (int i = 0; i < numBins - 1; i++) {
            if (leftCounts[i] == 0 || rightCounts[i] == 0) continue;

            float cost =
                1.0f + // traversal cost (Ct) - arbitrary scaling
                (leftCounts[i] * leftBoxes[i].GetSurfaceArea() +
                 rightCounts[i] * rightBoxes[i].GetSurfaceArea()) / parentArea;

            if (cost < bestCost) {
                bestCost = cost;
                bestAxis = axis;
                bestSplitBin = i;
            }
        }
    }

    // If SAH failed (all centroids collapsed), fallback to median split
    if (bestAxis == -1) {
        const size_t mid = (first + last) / 2;
        std::nth_element(work + first, work + mid, work + last,
            [](const BVH::Node& a, const BVH::Node& b) {
                AABB boxA = a.box;
                AABB boxB = b.box;
                return AABB::FindCentroid(boxA).x < AABB::FindCentroid(boxB).x;
            });
        const size_t leftIndex  = BuildHierarchyRecursively_SAH(outNodes, outCount, work, first, mid);
        const size_t rightIndex = BuildHierarchyRecursively_SAH(outNodes, outCount, work, mid,   last);
        AABB parentBox = AABB::UnionAABB(outNodes[leftIndex].box, outNodes[rightIndex].box);
        outNodes[outCount] = BVH::Node(leftIndex, rightIndex, parentBox);
        return outCount++;
    }

    // Partition primitives according to best split
    float cmin = bounds.lowerBound[bestAxis];
    float cmax = bounds.upperBound[bestAxis];
    float splitPos = cmin + (bestSplitBin + 1) * (cmax - cmin) / (float)numBins;

    auto midIter = std::partition(work + first, work + last,
        [bestAxis, splitPos](const BVH::Node& n) {
            AABB boxN = n.box;
            return AABB::FindCentroid(boxN)[bestAxis] < splitPos;
        });

    size_t mid = midIter - work;
    if (mid == first || mid == last) mid = (first + last) / 2; // Safety fallback

    // Recursively build children
    const size_t leftIndex  = BuildHierarchyRecursively_SAH(outNodes, outCount, work, first, mid);
    const size_t rightIndex = BuildHierarchyRecursively_SAH(outNodes, outCount, work, mid,   last);

    // Create parent node
    AABB parentBox = AABB::UnionAABB(outNodes[leftIndex].box, outNodes[rightIndex].box);
    outNodes[outCount] = BVH::Node(leftIndex, rightIndex, parentBox);
    return outCount++;
}
