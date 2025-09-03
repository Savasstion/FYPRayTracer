#include  "LightTree.cuh"

__host__ __device__ float ComputeClusterImportance(const LightTree::ShadingPointQuery& shadingPoint, const LightTree::Node& cluster)
{
    //  refer to the paper, Importance Sampling of Many Lights with Adaptive Tree Splitting at 5.1 Cluster Importance for Surfaces
    float theta_u;
    {
        ConeBounds cone = ConeBounds::FindConeThatEnvelopsAABBFromPoint(cluster.bounds_w, shadingPoint.position);
        theta_u = cone.theta_o;
    }

    glm::vec3 aabbPos;
    aabbPos.x = cluster.bounds_w.centroidPos.x;
    aabbPos.y = cluster.bounds_w.centroidPos.y;
    aabbPos.z = cluster.bounds_w.centroidPos.z;
    glm::vec3 dir = shadingPoint.position- aabbPos;
    float distanceSquared = fmaxf(glm::dot(dir, dir), 1e-12f);  //1e-12f is there to avoid zero division
    dir = glm::normalize(dir);
    
    float dotVal = glm::dot(cluster.bounds_o.axis, dir);
    dotVal = glm::clamp(dotVal, -1.0f, 1.0f);
    float theta = acosf(dotVal);

    float angleTerm = glm::clamp(theta - cluster.bounds_o.theta_o - theta_u, 0.0f, cluster.bounds_o.theta_e);
    float cosTerm = cosf(angleTerm);

    return (cluster.energy * cosTerm) / distanceSquared;
}

LightTree::SampledLight PickLight(const LightTree* tree, const LightTree::ShadingPointQuery& sp, uint32_t& randSeed)
{
    LightTree::SampledLight out;
    out.emitterIndex = static_cast<uint32_t>(-1);
    out.pmf = 0.0f;
    float randFloat = MathUtils::randomFloat(randSeed);

    if (!tree || tree->nodeCount == 0 || tree->rootIndex == static_cast<uint32_t>(-1)) {
        return out; // empty tree
    }

    // start at root
    uint32_t nodeIdx = tree->rootIndex;
    // accumulator of probability (product of branch choices)
    float pmfAcc = 1.0f;

    // Safety clamp u, prevent invalid array index
    randFloat = glm::clamp(randFloat, 0.0f, 0.9999999f);

    while (true) {
        const LightTree::Node& node = tree->nodes[nodeIdx];

        // Leaf: sample emitter from leaf
        if (node.isLeaf) {
            out.emitterIndex = node.emmiterIndex;   //  index to actual emmisive triangle
            out.pmf = pmfAcc;
            return out;
        }

        // Internal node: left child index is in offset, right child index stored in emmiterIndex
        uint32_t leftIdx  = node.offset;
        uint32_t rightIdx = node.emmiterIndex;
        
        // Compute importances of left and right child
        float I_left  = ComputeClusterImportance(sp, tree->nodes[leftIdx]);
        float I_right = ComputeClusterImportance(sp, tree->nodes[rightIdx]);
        
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

        // Decide branch using u, and remap u to conditional u' for the chosen branch:
        // if u < p_left -> go left and u' = u / p_left
        // else -> go right and u' = (u - p_left) / (1 - p_left)
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
