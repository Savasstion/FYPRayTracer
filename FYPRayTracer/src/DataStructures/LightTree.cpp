#include "LightTree.cuh"

void LightTree::AllocateNodes(size_t count)
{
    FreeNodes();
    nodes.reserve(2 * count - 1);    //  maximum amount of memory possibly needed
            
}

void LightTree::FreeNodes()
{
    if (!nodes.empty())
        nodes.clear();
}

float LightTree::GetOrientationBoundsAreaMeasure(const float& theta_o, const float& theta_e)
{
    constexpr float piHalf = 0.5f * MathUtils::pi;
    //  refer to 4.3 of Importance Sampling of Many Lights with Adaptive Tree Splitting by ALEJANDRO CONTYESTEVEZ

    float theta_w = fminf(theta_o + theta_e, MathUtils::pi);
    float a = (2 * MathUtils::pi) * (1 - glm::cos(theta_o));
    float b = piHalf * (2 * theta_w * glm::sin(theta_o) - glm::cos(theta_o - 2 * theta_w) - (2 * theta_o * glm::sin(theta_o)) + glm::cos(theta_o));

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
