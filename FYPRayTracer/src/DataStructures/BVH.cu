#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "BVH.cuh"
#include <algorithm>
#include "../Utility/MortonCode.cuh"
#include "../Utility/BitManipulation.cuh"

//  members for CUDA parallelization
namespace
{
    BVH::Node* d_ptr_nodes;
    BVH::Node* d_ptr_collisionObjects;
    BVH::MortonCodeEntry* h_ptr_sortedMortonCodes;
    BVH::MortonCodeEntry* d_ptr_sortedMortonCodes;
    AABB* d_ptr_objectAABBs;
    
    size_t nodeCount;
    bool isNewValuesSet = true;
    size_t objectCount = 0;
}

__host__ void BVH::CUDA_AllocateMemory(size_t currentObjCount)
{
    // if no increase in amount of space needed, then no need to allocate
    if(currentObjCount > objectCount)
    {
        nodeCount = 2 * currentObjCount - 1;
        cudaMalloc((void**)&d_ptr_nodes, nodeCount * sizeof(BVH::Node));
        cudaMalloc((void**)&d_ptr_collisionObjects, currentObjCount * sizeof(BVH::Node));

        h_ptr_sortedMortonCodes = (BVH::MortonCodeEntry*)malloc(nodeCount * sizeof(BVH::MortonCodeEntry));
        cudaMalloc((void**)&d_ptr_sortedMortonCodes, nodeCount * sizeof(BVH::MortonCodeEntry));
        cudaMalloc((void**)&d_ptr_objectAABBs, currentObjCount * sizeof(AABB));
        
        objectCount = currentObjCount;

    }

}

__host__ void BVH::CUDA_ConstructBVHInParallel(std::vector<Node>& objects)
{
    CUDA_ClearBVH();
    size_t objectCount = objects.size();
    CUDA_AllocateMemory(objectCount);

    //Build Hierarchy
    if (objectCount > 0)
    {
        rootIndex = CUDA_BuildHierarchyInParallel(objects.data(), objectCount);
        
    }
        
}

__host__ void BVH::CUDA_ClearBVH()
{
    nodes.clear();
    rootIndex = -1;
    
    CUDA_FreeDeviceSpaceForBVH();

}

__host__ void BVH::CUDA_FreeDeviceSpaceForBVH()
{
    cudaFree(d_ptr_nodes);
    cudaFree(d_ptr_collisionObjects);
    cudaFree(d_ptr_sortedMortonCodes);
    
    free(h_ptr_sortedMortonCodes);

    objectCount = 0;

    isNewValuesSet = true;
}

__host__ size_t BVH::CUDA_BuildHierarchyInParallel(Node* objects, size_t objectCount)
{
    //  set CUDA threads per block and amount of blocks
    size_t threadsPerBlock = 256;
    size_t leafBlocks = (objectCount + threadsPerBlock - 1) / threadsPerBlock;
    size_t internalBlocks = ((objectCount - 1) + threadsPerBlock - 1) / threadsPerBlock;
    
    CUDA_CopyComponentsFromHostToDevice(objects);
    
    // Assign Morton Codes
    CUDA_AssignMortonCodesKernel<<<leafBlocks, threadsPerBlock >>>(d_ptr_sortedMortonCodes, d_ptr_collisionObjects, objectCount, SceneSettings::minSceneBound, SceneSettings::maxSceneBound);
    cudaDeviceSynchronize();

    //  TODO : Actually implement a parallel binary radix sorting algorithm
    CUDA_SortMortonCodes(objectCount);
    //cudaDeviceSynchronize();
    
    //  first N nodes in p_ptr_nodes are leafs, the next N - 1 nodes will be internal nodes
    CUDA_BuildLeafNodesKernel<<<leafBlocks, threadsPerBlock>>>(d_ptr_sortedMortonCodes, d_ptr_nodes, d_ptr_objectAABBs, objectCount);
    cudaDeviceSynchronize();
    CUDA_BuildInternalNodesKernel<<<internalBlocks, threadsPerBlock>>>(d_ptr_sortedMortonCodes, d_ptr_nodes, objectCount);
    cudaDeviceSynchronize();

    CUDA_CopyDeviceNodesToHost();
    
    return objectCount; //root always N
}

__host__ void BVH::CUDA_SortMortonCodes(size_t objectCount)
{
    //  Take copy from device memory
    cudaMemcpy(h_ptr_sortedMortonCodes, d_ptr_sortedMortonCodes, objectCount * sizeof(MortonCodeEntry), cudaMemcpyDeviceToHost);
    
    std::sort(h_ptr_sortedMortonCodes, h_ptr_sortedMortonCodes + objectCount, [](const MortonCodeEntry& a, const MortonCodeEntry& b) {
        return a.mortonCode < b.mortonCode;
    });

    //  Copy back to device memory
    cudaMemcpy(d_ptr_sortedMortonCodes, h_ptr_sortedMortonCodes, objectCount * sizeof(MortonCodeEntry), cudaMemcpyHostToDevice);
}


__global__ void CUDA_AssignMortonCodesKernel(BVH::MortonCodeEntry* d_ptr_sortedMortonCodes, BVH::Node* d_ptr_collisionObjects,size_t objectCount, Vector3f minSceneBound, Vector3f maxSceneBound)
{
    //  Serial
    /*for(size_t i = 0 ; i < objectCount; i++)
    {
        d_ptr_sortedMortonCodes[i].objectIndex = i;

        auto eID = ptr_circle_collider_components[i].entityID;
        auto tID = ptr_entities[eID].transformComponentID;
        auto&& transform = ptr_transform_components[tID];
        d_ptr_sortedMortonCodes[i].mortonCode = morton2D(transform.position.x, transform.position.y);
    }*/

    //Parallel
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= objectCount) return;

    d_ptr_sortedMortonCodes[idx].objectIndex = idx;
    auto position = d_ptr_collisionObjects[idx].box.centroidPos;
    d_ptr_sortedMortonCodes[idx].mortonCode =
        morton3D(position.x, position.y, position.z, minSceneBound, maxSceneBound);
    
    
}

__global__ void CUDA_BuildLeafNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes, BVH::Node* ptr_nodes, AABB* ptr_objectAABBs, size_t objectCount)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= objectCount) return;

    size_t objIndex = ptr_sortedMortonCodes[idx].objectIndex;
    ptr_nodes[idx] = BVH::Node(objIndex, ptr_objectAABBs[objIndex]); // Leaf constructor
    
    // Serial version
    // first N indices will be leaf nodes
    //for (size_t idx = 0; idx < objectCount; idx++) // in parallel
    //{
    //    //leafNodes[idx].objectID = sortedObjectIDs[idx];
    //    p_ptr_nodes[idx] = Node(p_sortedMortonCodes[idx].objectIndex, objectAABBs[p_sortedMortonCodes[idx].objectIndex]);   //  Leaf node constructor
    //}
}

__global__ void CUDA_BuildInternalNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes, BVH::Node* ptr_nodes,
    size_t objectCount)
{
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= objectCount - 1) return;   
    
    int2 range = determineRange(ptr_sortedMortonCodes, objectCount, static_cast<int>(idx));
    int first = range.x;
    int last = range.y;

    //  get all encompassing leaf node's AABBs within the range
    AABB internalNodeBox = ptr_nodes[range.x].box;
    for (int i = range.x + 1; i <= range.y; i++)
    {
        if(ptr_nodes[i].isLeaf)
            internalNodeBox = AABB::UnionAABB(internalNodeBox, ptr_nodes[i].box);
    }
    
    int split = findSplit(ptr_sortedMortonCodes, first, last);

    size_t indexA = (split == first) ? split : objectCount + split;
    size_t indexB = (split + 1 == last) ? (split + 1) : objectCount + split + 1;

    ptr_nodes[idx + objectCount] = BVH::Node(indexA, indexB, internalNodeBox); // Internal constructor

    // Serial version
    //for (size_t idx = 0; idx < objectCount - 1; idx++) // in parallel
    //{
    //    // Find out which range of objects the node corresponds to.

    //    //  may need to convert idx to 32-bit int
    //    int2 range = determineRange(d_ptr_sortedMortonCodes, objectCount, idx);
    //    int first = range.x;
    //    int last = range.y;

    //    //get all encompassing leaf node's AABBs within the range
    // AABB internalNodeBox = ptr_nodes[range.x].box;
    // for (int i = range.x + 1; i <= range.y; i++)
    // {
    //     if(ptr_nodes[i].isLeaf)
    //         internalNodeBox = AABB::UnionAABB(internalNodeBox, ptr_nodes[i].box);
    // }

    //    // Determine where to split the range.
    //    int split = findSplit(d_ptr_sortedMortonCodes, first, last);

    //    // Select childA.
    //    size_t indexA;
    //    if (split == first)
    //    {
    //        indexA = split;
    //    }
    //    else
    //    {
    //        indexA = objectCount + split;
    //    }
    //    
    //    // Select childB.
    //    size_t indexB;
    //    if (split + 1 == last)
    //    {
    //        indexB = split + 1;
    //    }
    //    else
    //    {
    //        indexB = objectCount + split + 1;
    //    }
    //    
    //    // Record parent-child relationships.
    //    d_ptr_nodes[idx + objectCount] = Node(indexA, indexB, internalNodeBox);
    //}
}


__host__ void BVH::CUDA_CopyComponentsFromHostToDevice(BVH::Node* ptr_objects)
{
    //  If unchanged, no need to copy same values again
    if(isNewValuesSet)
    {
        cudaMemcpy(d_ptr_collisionObjects, ptr_objects, objectCount * sizeof(BVH::Node), cudaMemcpyHostToDevice);
        
        isNewValuesSet = false;
    }


    //get all AABBs of all objects, leaf nodes may be in different order every frame
    AABB* temp_ptr_objectAABBs = (AABB*)malloc(objectCount * sizeof(AABB));
    for (size_t i = 0; i < objectCount; i++)
    {
        temp_ptr_objectAABBs[i] = (ptr_objects[i].box);
    }
    cudaMemcpy(d_ptr_objectAABBs, temp_ptr_objectAABBs, objectCount * sizeof(AABB), cudaMemcpyHostToDevice);
    free(temp_ptr_objectAABBs);
    temp_ptr_objectAABBs = nullptr;
}

__host__ void BVH::CUDA_CopyDeviceNodesToHost()
{
    nodes.resize(2 * objectCount - 1); // allocate and set size
    cudaMemcpy(nodes.data(), d_ptr_nodes,
               (2 * objectCount - 1) * sizeof(Node),
               cudaMemcpyDeviceToHost);
}

__host__ __device__ int findSplit(BVH::MortonCodeEntry* morton, int first, int last)
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

__host__ __device__ int2 determineRange(BVH::MortonCodeEntry* p_sortedMortonCodes, int objectCount, int idx)
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