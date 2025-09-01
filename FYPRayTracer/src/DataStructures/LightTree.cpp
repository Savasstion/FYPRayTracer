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
