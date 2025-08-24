// #include "BVH.cuh"
// #include <algorithm>
// #include "../Utility/MortonCode.cuh"
//
// // OMP variable storage in anonymous namespace
// namespace
// {
//     std::vector<BVH::MortonCodeEntry> omp_sortedMortonCodes;
//     std::vector<AABB> omp_AABBs;
// }
//
// void BVH::OMP_ClearBVH()
// {
//     FreeHostNodes();  // free the allocated nodes array
//     omp_sortedMortonCodes.clear();
//     omp_AABBs.clear();
//     rootIndex = static_cast<size_t>(-1);
// }
//
// void BVH::OMP_ConstructBVHInParallel(Node* objects, size_t objectCount)
// {
//     OMP_ClearBVH();
//     if (objectCount > 0)
//     {
//         // Allocate host nodes: total 2 * N - 1
//         AllocateHostNodes(2 * objectCount - 1);
//
//         rootIndex = OMP_BuildHierarchyInParallel(objects, objectCount);
//     }
// }
//
// void BVH::OMP_AssignMortonCodes(size_t objectCount)
// {
//     #pragma omp parallel for
//     for (int i = 0; i < static_cast<int>(objectCount); i++)
//     {
//         omp_sortedMortonCodes[i].objectIndex = i;
//
//         float x = omp_AABBs[i].centroidPos.x;
//         float y = omp_AABBs[i].centroidPos.y;
//         float z = omp_AABBs[i].centroidPos.z;
//         omp_sortedMortonCodes[i].mortonCode = morton3D(x, y, z, SceneSettings::minSceneBound, SceneSettings::maxSceneBound);
//     }
// }
//
// void BVH::OMP_BuildLeafNodes(Node* ptr_nodes, size_t objectCount)
// {
//     #pragma omp parallel for
//     for (int idx = 0; idx < static_cast<int>(objectCount); idx++)
//     {
//         size_t objectIdx = omp_sortedMortonCodes[idx].objectIndex;
//         ptr_nodes[idx] = Node(objectIdx, omp_AABBs[objectIdx]);
//     }
// }
//
// void BVH::OMP_BuildInternalNodes(Node* ptr_nodes, size_t objectCount)
// {
//     #pragma omp parallel for
//     for (int idx = 0; idx < static_cast<int>(objectCount) - 1; idx++)
//     {
//         int2 range = determineRange(omp_sortedMortonCodes.data(), static_cast<int>(objectCount), idx);
//         int first = range.x;
//         int last = range.y;
//
//         AABB internalNodeBox = ptr_nodes[first].box;
//         for (int i = first + 1; i <= last; i++)
//         {
//             if (ptr_nodes[i].isLeaf)
//                 internalNodeBox = AABB::UnionAABB(internalNodeBox, ptr_nodes[i].box);
//         }
//
//         int split = findSplit(omp_sortedMortonCodes.data(), first, last);
//
//         size_t indexA = (split == first) ? split : objectCount + split;
//         size_t indexB = (split + 1 == last) ? split + 1 : objectCount + split + 1;
//
//         ptr_nodes[objectCount + idx] = Node(indexA, indexB, internalNodeBox);
//     }
// }
//
// size_t BVH::OMP_BuildHierarchyInParallel(Node* objects, size_t objectCount)
// {
//     omp_AABBs.resize(objectCount);
//     #pragma omp parallel for
//     for (int i = 0; i < static_cast<int>(objectCount); i++)
//     {
//         omp_AABBs[i] = objects[i].box;
//     }
//
//     omp_sortedMortonCodes.resize(objectCount);
//     OMP_AssignMortonCodes(objectCount);
//
//     std::sort(omp_sortedMortonCodes.begin(), omp_sortedMortonCodes.end(),
//         [](const MortonCodeEntry& a, const MortonCodeEntry& b) {
//             return a.mortonCode < b.mortonCode;
//         });
//
//     // Build leaf nodes
//     OMP_BuildLeafNodes(nodes, objectCount);
//
//     // Build internal nodes
//     OMP_BuildInternalNodes(nodes, objectCount);
//
//     // root is at index 2*N - 2 = nodesCount - 1 (last internal node)
//     return 2 * objectCount - 2;
// }
