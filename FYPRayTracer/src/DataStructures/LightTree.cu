#include "LightTree.cuh"
#include "../Classes/BaseClasses/Mesh_GPU.h"

__host__ __device__ LightTree::SampledLight PickLight_BLAS(const LightTree* blas_tree,
                                                           const LightTree::ShadingPointQuery& sp, float randFloat,
                                                           float currentPMF)
{
    LightTree::SampledLight out;
    out.pmf = 0.0f;
    out.emitterIndex = -1;

    if (!blas_tree || blas_tree->nodeCount == 0 || blas_tree->rootIndex == static_cast<uint32_t>(-1))
    {
        return out; // empty tree
    }

    // start at root
    uint32_t nodeIdx = blas_tree->rootIndex;
    // accumulator of probability (product of branch choices)
    float pmfAcc = 1.0f;

    // Safety clamp u, prevent invalid array index
    randFloat = glm::clamp(randFloat, 0.0f, 0.9999999f);

    while (true)
    {
        const LightTree::Node& node = blas_tree->nodes[nodeIdx];

        // Leaf: sample emitter from leaf
        if (node.isLeaf)
        {
            out.emitterIndex = node.emitterIndex; //  index to actual emissive triangle
            out.pmf = currentPMF * pmfAcc;
            return out;
        }

        // Internal node: left child index is in offset, right child index stored in emitterIndex
        uint32_t leftIdx = node.offset;
        uint32_t rightIdx = node.emitterIndex;

        // Compute importances of left and right child
        float I_left = ComputeClusterImportance(sp, blas_tree->nodes[leftIdx]);
        float I_right = ComputeClusterImportance(sp, blas_tree->nodes[rightIdx]);

        float sum = I_left + I_right;
        // Numeric guard
        if (!(sum > 0.0f) || (I_left + I_right) <= 0.0f)
        {
            // Unexpected; split evenly
            sum = 1.0f;
            I_left = 0.5f;
            //I_right = 0.5f;   (not needed, can skip)
        }

        float p_left = I_left / sum;
        // Clamp to avoid exact 0/1 which would break the interval mapping
        p_left = glm::clamp(p_left, 1e-6f, 1.0f - 1e-6f);

        if (randFloat < p_left)
        {
            // choose left
            pmfAcc *= p_left;
            // get new random number
            randFloat = randFloat / p_left; //  remap instead of complete resample random num (lesser variance maybe)
            nodeIdx = leftIdx;
        }
        else
        {
            // choose right
            float p_right = 1.0f - p_left;
            pmfAcc *= p_right;
            randFloat = (randFloat - p_left) / p_right;
            nodeIdx = rightIdx;
        }

        // loop continues until a leaf
    } // end while
}

__host__ __device__ LightTree::SampledLight PickLight_TLAS(const Mesh_GPU* meshes, const LightTree* tlas_tree,
                                                           const LightTree::ShadingPointQuery& sp, uint32_t& randSeed)
{
    LightTree::SampledLight out;
    out.emitterIndex = static_cast<uint32_t>(-1);
    out.pmf = 0.0f;
    float randFloat = MathUtils::randomFloat(randSeed);

    if (!tlas_tree || tlas_tree->nodeCount == 0 || tlas_tree->rootIndex == static_cast<uint32_t>(-1))
    {
        return out; // empty tree
    }

    // start at root
    uint32_t nodeIdx = tlas_tree->rootIndex;
    // accumulator of probability (product of branch choices)
    float pmfAcc = 1.0f;

    // Safety clamp u, prevent invalid array index
    randFloat = glm::clamp(randFloat, 0.0f, 0.9999999f);

    while (true)
    {
        const LightTree::Node& node = tlas_tree->nodes[nodeIdx];

        // Leaf: sample emitter from leaf
        if (node.isLeaf)
        {
            uint32_t blasIndex = node.emitterIndex; //  index to actual emmisive triangle
            out = PickLight_BLAS(meshes[blasIndex].lightTree_blas, sp, randFloat, pmfAcc);
            return out;
        }

        // Internal node: left child index is in offset, right child index stored in emmiterIndex
        uint32_t leftIdx = node.offset;
        uint32_t rightIdx = node.emitterIndex;

        // Compute importances of left and right child
        float I_left = ComputeClusterImportance(sp, tlas_tree->nodes[leftIdx]);
        float I_right = ComputeClusterImportance(sp, tlas_tree->nodes[rightIdx]);

        float sum = I_left + I_right;
        // Numeric guard
        if (!(sum > 0.0f) || (I_left + I_right) <= 0.0f)
        {
            // Unexpected; split evenly
            sum = 1.0f;
            I_left = 0.5f;
            //I_right = 0.5f;   (not needed, can skip)
        }

        float p_left = I_left / sum;
        // Clamp to avoid exact 0/1 which would break the interval mapping
        p_left = glm::clamp(p_left, 1e-6f, 1.0f - 1e-6f);

        if (randFloat < p_left)
        {
            // choose left
            pmfAcc *= p_left;
            // get new random number
            randFloat = randFloat / p_left; //  remap instead of complete resample random num (lesser variance maybe)
            nodeIdx = leftIdx;
        }
        else
        {
            // choose right
            float p_right = 1.0f - p_left;
            pmfAcc *= p_right;
            randFloat = (randFloat - p_left) / p_right;
            nodeIdx = rightIdx;
        }

        // loop continues until a leaf
    } // end while
}

__host__ __device__ uint32_t FindBLAS_LightTreeWithEmmiterIndexInTLAS_LightTree(
    const Mesh_GPU* meshes, const LightTree* tlas_tree, const uint32_t& indexToCheck)
{
    for (uint32_t i = 0; i < tlas_tree->nodeCount; i++)
    {
        if (!tlas_tree->nodes[i].isLeaf)
            // internal node, skip
            continue;


        if (FindNodeWithEmmiterIndexInLightTree(meshes[tlas_tree->nodes[i].emitterIndex].lightTree_blas->nodes,
                                                meshes[tlas_tree->nodes[i].emitterIndex].lightTree_blas->nodeCount,
                                                indexToCheck) != static_cast<uint32_t>(-1))
        {
            return i;
        }
    }
    return -1; // no leaf matched
}

__host__ __device__ uint32_t FindNodeWithEmmiterIndexInLightTree(LightTree::Node* nodes, const uint32_t& nodeCount,
                                                                 const uint32_t& indexToCheck)
{
    for (uint32_t i = 0; i < nodeCount; i++)
    {
        if (!nodes[i].isLeaf)
            // internal node, skip
            continue;

        if (nodes[i].emitterIndex == indexToCheck)
        {
            return i; // found a match in node i
        }
    }
    return -1; // no leaf matched
}

__host__ __device__ float ComputeDirectEmitterPMF(const Mesh_GPU* meshes, const LightTree* tlas_tree,
                                                  const LightTree::ShadingPointQuery& sp, uint32_t emitterIndex)
{
    if (!tlas_tree || tlas_tree->nodeCount == 0 || tlas_tree->rootIndex == static_cast<uint32_t>(-1))
        return 0.0f; // empty tree

    // 1) Find TLAS leaf whose BLAS contains the emitter
    uint32_t tlasLeafIdx = FindBLAS_LightTreeWithEmmiterIndexInTLAS_LightTree(meshes, tlas_tree, emitterIndex);
    if (tlasLeafIdx == static_cast<uint32_t>(-1))
        return 0.0f; // emitter not found

    // 2) Traverse TLAS from root to that leaf, accumulate PMF
    float pmfAcc = 1.0f;
    uint32_t nodeIdx = tlas_tree->rootIndex;

    while (!tlas_tree->nodes[nodeIdx].isLeaf)
    {
        uint32_t leftIdx = tlas_tree->nodes[nodeIdx].offset;
        uint32_t rightIdx = tlas_tree->nodes[nodeIdx].emitterIndex;

        float I_left = ComputeClusterImportance(sp, tlas_tree->nodes[leftIdx]);
        // pass dummy SP, or modify function to accept emitterIndex
        float I_right = ComputeClusterImportance(sp, tlas_tree->nodes[rightIdx]);

        float sum = I_left + I_right;
        if (!(sum > 0.0f))
        {
            I_left = 0.5f;
            sum = 1.0f;
        }

        float p_left = I_left / sum;

        // decide which branch contains the target leaf
        if (leftIdx <= tlasLeafIdx && tlasLeafIdx <= leftIdx + (tlas_tree->nodes[leftIdx].numEmitters - 1))
        {
            pmfAcc *= p_left;
            nodeIdx = leftIdx;
        }
        else
        {
            pmfAcc *= (1.0f - p_left);
            nodeIdx = rightIdx;
        }
    }

    // nodeIdx now points to the TLAS leaf
    uint32_t blasIndex = tlas_tree->nodes[nodeIdx].emitterIndex;
    LightTree* blasTree = meshes[blasIndex].lightTree_blas;
    if (!blasTree) return 0.0f;

    // 3) Traverse BLAS to the actual emitter leaf
    nodeIdx = blasTree->rootIndex;
    while (!blasTree->nodes[nodeIdx].isLeaf)
    {
        uint32_t leftIdx = blasTree->nodes[nodeIdx].offset;
        uint32_t rightIdx = blasTree->nodes[nodeIdx].emitterIndex;

        float I_left = ComputeClusterImportance(sp, blasTree->nodes[leftIdx]);
        float I_right = ComputeClusterImportance(sp, blasTree->nodes[rightIdx]);

        float sum = I_left + I_right;
        if (!(sum > 0.0f))
        {
            I_left = 0.5f;
            sum = 1.0f;
        }

        float p_left = I_left / sum;

        if (leftIdx <= emitterIndex && emitterIndex <= leftIdx + (blasTree->nodes[leftIdx].numEmitters - 1))
        {
            pmfAcc *= p_left;
            nodeIdx = leftIdx;
        }
        else
        {
            pmfAcc *= (1.0f - p_left);
            nodeIdx = rightIdx;
        }
    }

    return pmfAcc; // total probability of reaching that emitter via TLAS+BLAS
}
