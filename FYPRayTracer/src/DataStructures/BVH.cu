// #include "BVH.cuh"
// #include <cuda_runtime.h>
// #include <algorithm>
// #include "../Utility/MortonCode.cuh"
// #include "../Utility/BitManipulation.cuh"
// #include <iostream>
// #include <cfloat>
//
// //-----------------------------------
// // Memory allocation / free
// //-----------------------------------
// void BVH::CUDA_AllocateMemory(size_t currentObjCount)
// {
//     if (currentObjCount > objectCount)
//     {
//         nodeCount = 2 * currentObjCount - 1;
//
//         cudaError_t err;
//
//         err = cudaMalloc((void**)&d_ptr_nodes, nodeCount * sizeof(Node));
//         if (err != cudaSuccess) {
//             printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
//         }
//         
//         err = cudaMalloc((void**)&d_ptr_collisionObjects, currentObjCount * sizeof(Node));
//         if (err != cudaSuccess) {
//             printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
//         }
//         
//         h_ptr_sortedMortonCodes = (MortonCodeEntry*)malloc(currentObjCount * sizeof(MortonCodeEntry));
//         err = cudaMalloc((void**)&d_ptr_sortedMortonCodes, currentObjCount * sizeof(MortonCodeEntry));
//         if (err != cudaSuccess) {
//             printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
//         }
//         
//         err = cudaMalloc((void**)&d_ptr_objectAABBs, currentObjCount * sizeof(AABB));
//         if (err != cudaSuccess) {
//             printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
//         }
//
//         objectCount = currentObjCount;
//     }
// }
//
// void BVH::CUDA_ClearBVH()
// {
//     FreeHostNodes();
//     rootIndex = static_cast<size_t>(-1);
//     CUDA_FreeDeviceSpaceForBVH();
// }
//
// void BVH::CUDA_FreeDeviceSpaceForBVH()
// {
//     cudaFree(d_ptr_nodes);
//     cudaFree(d_ptr_collisionObjects);
//     cudaFree(d_ptr_sortedMortonCodes);
//     cudaFree(d_ptr_objectAABBs);
//
//     if (h_ptr_sortedMortonCodes)
//     {
//         free(h_ptr_sortedMortonCodes);
//         h_ptr_sortedMortonCodes = nullptr;
//     }
//
//     objectCount = 0;
//     isNewValuesSet = true;
// }
//
// //-----------------------------------
// // BVH Construction
// //-----------------------------------
// void BVH::CUDA_ConstructBVHInParallel(Node* objects, size_t objCount)
// {
//     CUDA_ClearBVH();
//     CUDA_AllocateMemory(objCount);
//
//     if (objCount > 0)
//     {
//         rootIndex = CUDA_BuildHierarchyInParallel(objects, objCount);
//     }
// }
//
// size_t BVH::CUDA_BuildHierarchyInParallel(Node* objects, size_t objCount)
// {
//     size_t threadsPerBlock = 256;
//     size_t leafBlocks = (objCount + threadsPerBlock - 1) / threadsPerBlock;
//     size_t internalBlocks = ((objCount - 1) + threadsPerBlock - 1) / threadsPerBlock;
//
//     CUDA_CopyComponentsFromHostToDevice(objects);
//
//     CUDA_AssignMortonCodesKernel<<<leafBlocks, threadsPerBlock>>> (
//         d_ptr_sortedMortonCodes, d_ptr_collisionObjects, objCount,
//         SceneSettings::minSceneBound, SceneSettings::maxSceneBound);
//     cudaDeviceSynchronize();
//
//     CUDA_SortMortonCodes(objCount);
//
//     CUDA_BuildLeafNodesKernel<<<leafBlocks, threadsPerBlock>>> (
//         d_ptr_sortedMortonCodes, d_ptr_nodes, d_ptr_objectAABBs, objCount, objectOffset);
//     cudaDeviceSynchronize();
//
//     CUDA_BuildInternalNodesKernel<<<internalBlocks, threadsPerBlock>>> (
//         d_ptr_sortedMortonCodes, d_ptr_nodes, objCount);
//     cudaDeviceSynchronize();
//
//     CUDA_CopyDeviceNodesToHost();
//
//     cudaError_t err;
//     err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("cuda shit failed: %s\n", cudaGetErrorString(err));
//     }
//
//     return objCount; // root always at index N
// }
//
// void BVH::CUDA_SortMortonCodes(size_t objCount)
// {
//     cudaMemcpy(h_ptr_sortedMortonCodes, d_ptr_sortedMortonCodes, objCount * sizeof(MortonCodeEntry), cudaMemcpyDeviceToHost);
//
//     std::sort(h_ptr_sortedMortonCodes, h_ptr_sortedMortonCodes + objCount,
//         [](const MortonCodeEntry& a, const MortonCodeEntry& b) { return a.mortonCode < b.mortonCode; });
//
//     cudaMemcpy(d_ptr_sortedMortonCodes, h_ptr_sortedMortonCodes, objCount * sizeof(MortonCodeEntry), cudaMemcpyHostToDevice);
// }
//
// //-----------------------------------
// // Copy / sync
// //-----------------------------------
// void BVH::CUDA_CopyComponentsFromHostToDevice(Node* ptr_objects)
// {
//     if (isNewValuesSet)
//     {
//         cudaMemcpy(d_ptr_collisionObjects, ptr_objects, objectCount * sizeof(Node), cudaMemcpyHostToDevice);
//         isNewValuesSet = false;
//     }
//
//     // Copy AABBs
//     AABB* tempAABBs = (AABB*)malloc(objectCount * sizeof(AABB));
//     for (size_t i = 0; i < objectCount; i++)
//         tempAABBs[i] = ptr_objects[i].box;
//
//     cudaMemcpy(d_ptr_objectAABBs, tempAABBs, objectCount * sizeof(AABB), cudaMemcpyHostToDevice);
//     free(tempAABBs);
// }
//
// void BVH::CUDA_CopyDeviceNodesToHost()
// {
//     AllocateHostNodes(nodeCount); // ensure proper host allocation
//     cudaMemcpy(nodes, d_ptr_nodes, nodeCount * sizeof(Node), cudaMemcpyDeviceToHost);
// }
//
// //-----------------------------------
// // Device Kernels
// //-----------------------------------
// __global__ void CUDA_AssignMortonCodesKernel(BVH::MortonCodeEntry* d_ptr_sortedMortonCodes,
//     BVH::Node* d_ptr_collisionObjects,
//     size_t objectCount,
//     Vector3f minSceneBound, Vector3f maxSceneBound)
// {
//     size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx >= objectCount) return;
//
//     d_ptr_sortedMortonCodes[idx].objectIndex = d_ptr_collisionObjects[idx].objectIndex;
//     Vector3f pos = d_ptr_collisionObjects[idx].box.centroidPos;
//     d_ptr_sortedMortonCodes[idx].mortonCode = morton3D(pos.x, pos.y, pos.z, minSceneBound, maxSceneBound);
// }
//
// __global__ void CUDA_BuildLeafNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes,
//     BVH::Node* ptr_nodes,
//     AABB* ptr_objectAABBs,
//     size_t objectCount,
//     size_t objectOffset)
// {
//     size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx >= objectCount) return;
//
//     size_t objIndex = ptr_sortedMortonCodes[idx].objectIndex;
//     ptr_nodes[idx] = BVH::Node(objIndex, ptr_objectAABBs[objIndex - objectOffset]);
// }
//
// __global__ void CUDA_BuildInternalNodesKernel(BVH::MortonCodeEntry* ptr_sortedMortonCodes,
//     BVH::Node* ptr_nodes,
//     size_t objCount)
// {
//     size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
//     if (idx >= objCount - 1) return;
//
//     int2 range = determineRange(ptr_sortedMortonCodes, objCount, idx);
//     int first = range.x;
//     int last = range.y;
//
//     AABB internalBox = ptr_nodes[first].box;
//     for (int i = first + 1; i <= last; i++)
//     {
//         if (ptr_nodes[i].isLeaf)
//             internalBox = AABB::UnionAABB(internalBox, ptr_nodes[i].box);
//     }
//
//     int split = findSplit(ptr_sortedMortonCodes, first, last);
//
//     size_t indexA = (split == first) ? split : objCount + split;
//     size_t indexB = (split + 1 == last) ? (split + 1) : objCount + split + 1;
//
//     ptr_nodes[objCount + idx] = BVH::Node(indexA, indexB, internalBox);
// }
//
// //-----------------------------------
// // Utility functions
// //-----------------------------------
// int findSplit(BVH::MortonCodeEntry* morton, int first, int last)
// {
//     unsigned int firstCode = morton[first].mortonCode;
//     unsigned int lastCode = morton[last].mortonCode;
//     if (firstCode == lastCode) return (first + last) >> 1;
//
//     int delta_node = clz(firstCode ^ lastCode);
//     int split = first;
//     int stride = last - first;
//
//     do
//     {
//         stride = (stride + 1) >> 1;
//         int middle = split + stride;
//         if (middle < last)
//         {
//             int delta = clz(firstCode ^ morton[middle].mortonCode);
//             if (delta > delta_node) split = middle;
//         }
//     } while (stride > 1);
//
//     return split;
// }
//
// __host__ __device__ int2 determineRange(BVH::MortonCodeEntry* p_sortedMortonCodes, int objectCount, int idx)
// {
//     if (idx == 0) return make_int2(0, objectCount - 1);
//
//     unsigned int selfCode = p_sortedMortonCodes[idx].mortonCode;
//     int deltaL = clz(selfCode ^ p_sortedMortonCodes[idx - 1].mortonCode);
//     int deltaR = clz(selfCode ^ p_sortedMortonCodes[idx + 1].mortonCode);
//     int direction = (deltaR > deltaL) ? 1 : -1;
//     int deltaMin = fminf(deltaL, deltaR);
//
//     int lmax = 2;
//     int delta = -1;
//     int i_tmp = idx + direction * lmax;
//     if (0 <= i_tmp && i_tmp < objectCount) delta = clz(selfCode ^ p_sortedMortonCodes[i_tmp].mortonCode);
//
//     while (delta > deltaMin)
//     {
//         lmax <<= 1;
//         i_tmp = idx + direction * lmax;
//         delta = 1;
//         if (0 <= i_tmp && i_tmp < objectCount) delta = clz(selfCode ^ p_sortedMortonCodes[i_tmp].mortonCode);
//     }
//
//     int l = 0;
//     int t = lmax >> 1;
//     while (t > 0)
//     {
//         i_tmp = idx + (l + t) * direction;
//         delta = -1;
//         if (0 <= i_tmp && i_tmp < objectCount) delta = clz(selfCode ^ p_sortedMortonCodes[i_tmp].mortonCode);
//         if (delta > deltaMin) l += t;
//         t >>= 1;
//     }
//
//     int jdx = idx + l * direction;
//     if (direction < 0)
//     {
//         //std::swap(idx, jdx);
//         auto tmp = idx;
//         idx = jdx;
//         jdx = tmp;
//     } 
//
//     return make_int2(idx, jdx);
// }
