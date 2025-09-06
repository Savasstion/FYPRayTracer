#include "LightTree.cuh"
#include "../Classes/BaseClasses/Mesh_GPU.h"

__host__ __device__ LightTree::SampledLight PickLight_BLAS(const LightTree* blas_tree, const LightTree::ShadingPointQuery& sp, uint32_t& randSeed, LightTree::SampledLight currentSampledLight)
{
    float randFloat = MathUtils::randomFloat(randSeed);

    if (!blas_tree || blas_tree->nodeCount == 0 || blas_tree->rootIndex == static_cast<uint32_t>(-1)) {
        return currentSampledLight; // empty tree
    }

    // start at root
    uint32_t nodeIdx = blas_tree->rootIndex;
    // accumulator of probability (product of branch choices)
    float pmfAcc = currentSampledLight.pmf;

    // Safety clamp u, prevent invalid array index
    randFloat = glm::clamp(randFloat, 0.0f, 0.9999999f);

    while (true) {
        const LightTree::Node& node = blas_tree->nodes[nodeIdx];

        // Leaf: sample emitter from leaf
        if (node.isLeaf) {
            currentSampledLight.emitterIndex = node.emmiterIndex;   //  index to actual emmisive triangle
            currentSampledLight.pmf = pmfAcc;
            return currentSampledLight;
        }

        // Internal node: left child index is in offset, right child index stored in emmiterIndex
        uint32_t leftIdx  = node.offset;
        uint32_t rightIdx = node.emmiterIndex;
        
        // Compute importances of left and right child
        float I_left  = ComputeClusterImportance(sp, blas_tree->nodes[leftIdx]);
        float I_right = ComputeClusterImportance(sp, blas_tree->nodes[rightIdx]);
        
        float sum = I_left + I_right;
        // Numeric guard
        if (!(sum > 0.0f) || (I_left + I_right) <= 0.0f) {
            // Unexpected; split evenly
            sum = 1.0f;
            I_left = 0.5f;
            //I_right = 0.5f;   (not needed, can skip)
        }

        float p_left = I_left / sum;
        // Clamp to avoid exact 0/1 which would break the interval mapping
        p_left = glm::clamp(p_left, 1e-6f, 1.0f - 1e-6f);
        
        if (randFloat < p_left) {
            // choose left
            pmfAcc *= p_left;
            // get new random number
            randFloat = randFloat / p_left; //  remap instead of complete resample random num (lesser variance maybe)
            nodeIdx = leftIdx;
        } else {
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

    if (!tlas_tree || tlas_tree->nodeCount == 0 || tlas_tree->rootIndex == static_cast<uint32_t>(-1)) {
        return out; // empty tree
    }

    // start at root
    uint32_t nodeIdx = tlas_tree->rootIndex;
    // accumulator of probability (product of branch choices)
    float pmfAcc = 1.0f;

    // Safety clamp u, prevent invalid array index
    randFloat = glm::clamp(randFloat, 0.0f, 0.9999999f);

    while (true) {
        const LightTree::Node& node = tlas_tree->nodes[nodeIdx];

        // Leaf: sample emitter from leaf
        if (node.isLeaf) {
            uint32_t blasIndex = node.emmiterIndex;   //  index to actual emmisive triangle
            out.pmf = pmfAcc;
            
            out = PickLight_BLAS(meshes[blasIndex].lightTree_blas, sp, randSeed, out);
            return out;
        }

        // Internal node: left child index is in offset, right child index stored in emmiterIndex
        uint32_t leftIdx  = node.offset;
        uint32_t rightIdx = node.emmiterIndex;
        
        // Compute importances of left and right child
        float I_left  = ComputeClusterImportance(sp, tlas_tree->nodes[leftIdx]);
        float I_right = ComputeClusterImportance(sp, tlas_tree->nodes[rightIdx]);
        
        float sum = I_left + I_right;
        // Numeric guard
        if (!(sum > 0.0f) || (I_left + I_right) <= 0.0f) {
            // Unexpected; split evenly
            sum = 1.0f;
            I_left = 0.5f;
            //I_right = 0.5f;   (not needed, can skip)
        }

        float p_left = I_left / sum;
        // Clamp to avoid exact 0/1 which would break the interval mapping
        p_left = glm::clamp(p_left, 1e-6f, 1.0f - 1e-6f);
        
        if (randFloat < p_left) {
            // choose left
            pmfAcc *= p_left;
            // get new random number
            randFloat = randFloat / p_left; //  remap instead of complete resample random num (lesser variance maybe)
            nodeIdx = leftIdx;
        } else {
            // choose right
            float p_right = 1.0f - p_left;
            pmfAcc *= p_right;
            randFloat = (randFloat - p_left) / p_right;
            nodeIdx = rightIdx;
        }

        // loop continues until a leaf
    } // end while
}
