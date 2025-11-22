#include "LightTree.cuh"
#include <algorithm>

void LightTree::ConstructLightTree(Node* objects, uint32_t objCount)
{
    ClearLightTree();
    nodeCount = objCount;

    if (nodeCount > 0)
    {
        // Allocate host nodes: total 2 * N - 1 (max possibly needed)
        AllocateNodes(2 * nodeCount - 1);

        uint32_t outCount = 0;

        rootIndex = BuildHierarchyRecursively_SAOH(nodes, outCount, objects, 0, nodeCount);
        nodeCount = outCount;
    }
}

uint32_t LightTree::BuildHierarchyRecursively_SAOH(Node* outNodes, uint32_t& outCount, Node* work, uint32_t first,
                                                   uint32_t last)
{
    const uint32_t count = last - first;

    // Leaf node
    if (count == 1)
    {
        // copy emitter as leaf
        outNodes[outCount] = Node(work[first].emitterIndex, work[first].position, work[first].bounds_w,
                                  work[first].bounds_o, work[first].energy);
        return outCount++;
    }

    // Compute spatial bounds for current range (parent AABB)
    AABB parentBounds = work[first].bounds_w;
    for (uint32_t i = first + 1; i < last; ++i)
        parentBounds = AABB::UnionAABB(parentBounds, work[i].bounds_w);

    // Compute parent orientation cone (conservative union) and total energy
    ConeBounds parentCone = work[first].bounds_o;
    float parentEnergy = work[first].energy;
    for (uint32_t i = first + 1; i < last; ++i)
    {
        parentCone = ConeBounds::UnionCone(parentCone, work[i].bounds_o);
        parentEnergy += work[i].energy;
    }

    const float parentArea = parentBounds.GetSurfaceArea();
    const float parentOrientMeasure = GetOrientationBoundsAreaMeasure(parentCone.theta_o, parentCone.theta_e);
    float parentProb = GetProbabilityOfSamplingCluster(parentArea, parentOrientMeasure, parentEnergy);
    if (parentProb <= 0.0f) parentProb = 1e-12f; // guard to not divide by zero

    // SAOH parameters
    const int numBins = 16; // 8 - 32 gives good result
    float bestCost = FLT_MAX;
    int bestAxis = -1;
    int bestSplitBin = -1;

    // Try splitting along each axis
    for (int axis = 0; axis < 3; ++axis)
    {
        // Compute centroid bounds along this axis (we use Node.position as the centroid)
        float cmin = FLT_MAX, cmax = -FLT_MAX;
        for (uint32_t i = first; i < last; ++i)
        {
            float v = work[i].position[axis];
            if (v < cmin) cmin = v;
            if (v > cmax) cmax = v;
        }
        if (cmin == cmax) continue; // Degenerate axis → skip

        // Initialize bins
        Bin bins[numBins];
        for (int b = 0; b < numBins; ++b) bins[b].Reset();

        // Fill bins
        const float invRange = 1.0f / (cmax - cmin);
        for (uint32_t i = first; i < last; ++i)
        {
            float v = work[i].position[axis];
            int idx = (int)(((v - cmin) * invRange) * (numBins - 1));
            idx = glm::clamp(idx, 0, numBins - 1);
            bins[idx].AddEmitter(work[i]);
        }

        // Prefix sums (left cumulative)
        AABB leftBoxes[numBins - 1];
        ConeBounds leftCones[numBins - 1];
        float leftEnergy[numBins - 1];
        uint32_t leftCounts[numBins - 1];

        {
            bool any = false;
            AABB curAABB;
            ConeBounds curCone;
            float curEnergy = 0.0f;
            uint32_t curCount = 0;

            for (int i = 0; i < numBins - 1; ++i)
            {
                if (!any && bins[i].numEmitters == 0)
                {
                    // leave defaults (empty)
                    leftBoxes[i] = curAABB;
                    leftCones[i] = curCone;
                    leftEnergy[i] = curEnergy;
                    leftCounts[i] = curCount;
                    continue;
                }
                if (!any && bins[i].numEmitters > 0)
                {
                    curAABB = bins[i].bounds_w;
                    curCone = bins[i].bounds_o;
                    curEnergy = bins[i].energy;
                    curCount = bins[i].numEmitters;
                    any = true;
                }
                else if (bins[i].numEmitters > 0)
                {
                    curAABB = AABB::UnionAABB(curAABB, bins[i].bounds_w);
                    curCone = ConeBounds::UnionCone(curCone, bins[i].bounds_o);
                    curEnergy += bins[i].energy;
                    curCount += bins[i].numEmitters;
                }
                leftBoxes[i] = curAABB;
                leftCones[i] = curCone;
                leftEnergy[i] = curEnergy;
                leftCounts[i] = curCount;
            }
        }

        // Suffix sums (right cumulative)
        AABB rightBoxes[numBins - 1];
        ConeBounds rightCones[numBins - 1];
        float rightEnergy[numBins - 1];
        uint32_t rightCounts[numBins - 1];

        {
            bool any = false;
            AABB curAABB;
            ConeBounds curCone;
            float curEnergy = 0.0f;
            uint32_t curCount = 0;

            for (int i = numBins - 1; i > 0; --i)
            {
                int idx = i;
                if (!any && bins[idx].numEmitters == 0)
                {
                    rightBoxes[i - 1] = curAABB;
                    rightCones[i - 1] = curCone;
                    rightEnergy[i - 1] = curEnergy;
                    rightCounts[i - 1] = curCount;
                    continue;
                }
                if (!any && bins[idx].numEmitters > 0)
                {
                    curAABB = bins[idx].bounds_w;
                    curCone = bins[idx].bounds_o;
                    curEnergy = bins[idx].energy;
                    curCount = bins[idx].numEmitters;
                    any = true;
                }
                else if (bins[idx].numEmitters > 0)
                {
                    curAABB = AABB::UnionAABB(curAABB, bins[idx].bounds_w);
                    curCone = ConeBounds::UnionCone(curCone, bins[idx].bounds_o);
                    curEnergy += bins[idx].energy;
                    curCount += bins[idx].numEmitters;
                }
                rightBoxes[i - 1] = curAABB;
                rightCones[i - 1] = curCone;
                rightEnergy[i - 1] = curEnergy;
                rightCounts[i - 1] = curCount;
            }
        }

        // Evaluate costs
        for (int i = 0; i < numBins - 1; ++i)
        {
            if (leftCounts[i] == 0 || rightCounts[i] == 0) continue;

            float MA_left = leftBoxes[i].GetSurfaceArea();
            float M_omega_left = GetOrientationBoundsAreaMeasure(leftCones[i].theta_o, leftCones[i].theta_e);
            float E_left = leftEnergy[i];
            float P_left = GetProbabilityOfSamplingCluster(MA_left, M_omega_left, E_left);

            float MA_right = rightBoxes[i].GetSurfaceArea();
            float M_omega_right = GetOrientationBoundsAreaMeasure(rightCones[i].theta_o, rightCones[i].theta_e);
            float E_right = rightEnergy[i];
            float P_right = GetProbabilityOfSamplingCluster(MA_right, M_omega_right, E_right);

            float cost = GetSplitCost(P_left, P_right, parentProb);

            // determine regularizer
            float lengthMax = parentBounds.upperBound.x - parentBounds.lowerBound.x;
            lengthMax = glm::max(lengthMax, parentBounds.upperBound.y - parentBounds.lowerBound.y);
            lengthMax = glm::max(lengthMax, parentBounds.upperBound.z - parentBounds.lowerBound.z);
            lengthMax = glm::max(lengthMax, 1e-12f); // guard to prevent zero division

            float leftLen = leftBoxes[i].upperBound[bestAxis] - leftBoxes[i].lowerBound[bestAxis];
            float rightLen = rightBoxes[i].upperBound[bestAxis] - rightBoxes[i].lowerBound[bestAxis];
            leftLen = glm::max(leftLen, 1e-12f);
            rightLen = glm::max(rightLen, 1e-12f);

            float kr_left = lengthMax / leftLen;
            float kr_right = lengthMax / rightLen;
            float kr = glm::max(kr_left, kr_right);
            if (kr < 1.0f)
                kr = 1.0f;

            cost *= kr;
            // apply regularizer (avoids thin boxes, good for stratification according to the slides from Semantic Scholar about the paper)

            if (cost < bestCost)
            {
                bestCost = cost;
                bestAxis = axis;
                bestSplitBin = i;
            }
        }
    } // end axis loop

    // If SAOH failed (all centroids collapsed), fallback to median split
    if (bestAxis == -1)
    {
        const uint32_t mid = (first + last) / 2;
        std::nth_element(work + first, work + mid, work + last,
                         [](const Node& a, const Node& b)
                         {
                             return a.position.x < b.position.x;
                         });
        const uint32_t leftIndex = BuildHierarchyRecursively_SAOH(outNodes, outCount, work, first, mid);
        const uint32_t rightIndex = BuildHierarchyRecursively_SAOH(outNodes, outCount, work, mid, last);

        AABB parentBox = AABB::UnionAABB(outNodes[leftIndex].bounds_w, outNodes[rightIndex].bounds_w);
        ConeBounds parentNodeCone = ConeBounds::UnionCone(outNodes[leftIndex].bounds_o, outNodes[rightIndex].bounds_o);
        float parentNodeEnergy = outNodes[leftIndex].energy + outNodes[rightIndex].energy;

        //  parent node
        outNodes[outCount] = Node();
        outNodes[outCount].isLeaf = false;
        outNodes[outCount].offset = leftIndex;
        outNodes[outCount].emitterIndex = rightIndex;
        outNodes[outCount].bounds_w = parentBox;
        outNodes[outCount].bounds_o = parentNodeCone;
        outNodes[outCount].energy = parentNodeEnergy;
        outNodes[outCount].numEmitters = outNodes[leftIndex].numEmitters + outNodes[rightIndex].numEmitters;

        return outCount++;
    }

    // Partition primitives according to best split
    // recompute centroid extent for chosen axis
    float pmin = FLT_MAX, pmax = -FLT_MAX;
    for (uint32_t i = first; i < last; ++i)
    {
        float v = work[i].position[bestAxis];
        if (v < pmin) pmin = v;
        if (v > pmax) pmax = v;
    }
    float splitPos = pmin + (bestSplitBin + 1) * (pmax - pmin) / (float)numBins;

    auto midIter = std::partition(work + first, work + last,
                                  [bestAxis, splitPos](const Node& n)
                                  {
                                      return n.position[bestAxis] < splitPos;
                                  });

    uint32_t mid = (uint32_t)(midIter - work);
    if (mid == first || mid == last) mid = (first + last) / 2; // Safety fallback

    // Recursively build children
    const uint32_t leftIndex = BuildHierarchyRecursively_SAOH(outNodes, outCount, work, first, mid);
    const uint32_t rightIndex = BuildHierarchyRecursively_SAOH(outNodes, outCount, work, mid, last);

    // Create parent node
    AABB parentBox = AABB::UnionAABB(outNodes[leftIndex].bounds_w, outNodes[rightIndex].bounds_w);
    ConeBounds parentNodeCone = ConeBounds::UnionCone(outNodes[leftIndex].bounds_o, outNodes[rightIndex].bounds_o);
    float parentNodeEnergy = outNodes[leftIndex].energy + outNodes[rightIndex].energy;

    outNodes[outCount] = Node();
    outNodes[outCount].isLeaf = false;
    outNodes[outCount].offset = leftIndex;
    outNodes[outCount].emitterIndex = rightIndex;
    outNodes[outCount].bounds_w = parentBox;
    outNodes[outCount].bounds_o = parentNodeCone;
    outNodes[outCount].energy = parentNodeEnergy;
    outNodes[outCount].numEmitters = outNodes[leftIndex].numEmitters + outNodes[rightIndex].numEmitters;

    return outCount++;
}

void LightTree::ClearLightTree()
{
    FreeNodes(); // free the allocated nodes array
    rootIndex = static_cast<uint32_t>(-1);
}

void LightTree::AllocateNodes(uint32_t count)
{
    FreeNodes();
    nodes = new Node[count];
}

void LightTree::FreeNodes()
{
    if (nodes != nullptr)
    {
        delete[] nodes;
        nodeCount = 0;
    }

    nodes = nullptr;
}

float LightTree::GetOrientationBoundsAreaMeasure(float theta_o, float theta_e)
{
    constexpr float piHalf = 0.5f * MathUtils::pi;
    //  refer to 4.3 of Importance Sampling of Many Lights with Adaptive Tree Splitting by ALEJANDRO CONTYESTEVEZ

    float theta_w = fminf(theta_o + theta_e, MathUtils::pi);
    float a = (2 * MathUtils::pi) * (1 - glm::cos(theta_o));
    float b = piHalf * (2 * theta_w * glm::sin(theta_o) - glm::cos(theta_o - 2 * theta_w) - (2 * theta_o *
        glm::sin(theta_o)) + glm::cos(theta_o));

    return a + b;
}

float LightTree::GetProbabilityOfSamplingCluster(float area, float orientBoundAreaMeasure, float energy)
{
    // MA(C) * MΩ(C) * Energy(C)
    return area * orientBoundAreaMeasure * energy;
}

float LightTree::GetSplitCost(float probLeftCluster, float probRightCluster, float probCluster)
{
    return (probLeftCluster + probRightCluster) / probCluster;
}
