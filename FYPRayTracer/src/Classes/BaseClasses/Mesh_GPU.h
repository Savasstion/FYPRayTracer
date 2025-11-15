#ifndef MESH_GPU_H
#define MESH_GPU_H
#include "Mesh.h"

struct Mesh_GPU
{
    glm::vec3 position{0, 0, 0};
    glm::vec3 rotation{0, 0, 0};
    glm::vec3 scale{1, 1, 1};
    glm::mat4 worldTransformMatrix;
    bool isTransformed = false;
    //  if any of the transform data is modified, this should be true and then only need to do calculations to retransform the object 

    uint32_t vertexStart;
    uint32_t vertexCount;
    uint32_t indexStart;
    uint32_t indexCount;

    int materialIndex;

    AABB aabb;
    BVH* blas;
    LightTree* lightTree_blas;
};

inline Mesh_GPU MeshToHostMeshGPU(const Mesh& h_mesh)
{
    Mesh_GPU h_gpuMesh{};

    // Basic members copying
    h_gpuMesh.position = h_mesh.position;
    h_gpuMesh.rotation = h_mesh.rotation;
    h_gpuMesh.scale = h_mesh.scale;
    h_gpuMesh.worldTransformMatrix = h_mesh.worldTransformMatrix;
    h_gpuMesh.isTransformed = h_mesh.isTransformed;

    h_gpuMesh.vertexStart = h_mesh.vertexStart;
    h_gpuMesh.vertexCount = h_mesh.vertexCount;
    h_gpuMesh.indexStart = h_mesh.indexStart;
    h_gpuMesh.indexCount = h_mesh.indexCount;

    h_gpuMesh.materialIndex = h_mesh.materialIndex;

    h_gpuMesh.aabb = h_mesh.aabb;

    if (h_mesh.blas.nodeCount > 0)
        h_gpuMesh.blas = BVHToGPU(h_mesh.blas);

    if (h_mesh.lightTree_blas.nodeCount > 0)
        h_gpuMesh.lightTree_blas = LightTreeToGPU(h_mesh.lightTree_blas);

    return h_gpuMesh;
}

inline void FreeMeshGPU(Mesh_GPU* d_mesh)
{
    if (!d_mesh) return;

    cudaError_t err;

    // Copy back to host to inspect BLAS pointer
    Mesh_GPU h_mesh{};
    err = cudaMemcpy(&h_mesh, d_mesh, sizeof(Mesh_GPU), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        std::cerr << "cudaMemcpy failed\n";

    // Free BLAS
    if (h_mesh.blas)
        FreeBVH_GPU(h_mesh.blas);

    // Free Light Tree
    if (h_mesh.lightTree_blas)
        FreeLightTree_GPU(h_mesh.lightTree_blas);

    // Free the mesh
    err = cudaFree(d_mesh);
    if (err != cudaSuccess)
        std::cerr << "cudaFree failed\n";
}


#endif
