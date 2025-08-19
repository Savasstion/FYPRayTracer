#ifndef MESH_GPU_H
#define MESH_GPU_H
#include "Mesh.h"

struct Mesh_GPU
{
    glm::vec3 position{0,0,0};
    glm::vec3 rotation{0,0,0};
    glm::vec3 scale{1,1,1};
    glm::mat4 worldTransformMatrix;
    bool isTransformed = false; //  if any of the transform data is modified, this should be true and then only need to do calculations to retransform the object 
    
    uint32_t vertexStart;
    uint32_t vertexCount;
    uint32_t indexStart;
    uint32_t indexCount;
    
    int materialIndex;

    AABB aabb;
    BVH* blas;
};

__host__ __forceinline__ Mesh_GPU* MeshToGPU(const Mesh& h_mesh)
{
    Mesh_GPU h_gpuMesh{};

    // Basic members copying
    h_gpuMesh.position                 = h_mesh.position;
    h_gpuMesh.rotation                 = h_mesh.rotation;
    h_gpuMesh.scale                    = h_mesh.scale;
    h_gpuMesh.worldTransformMatrix     = h_mesh.worldTransformMatrix;
    h_gpuMesh.isTransformed            = h_mesh.isTransformed;
    
    h_gpuMesh.vertexStart      = h_mesh.vertexStart;
    h_gpuMesh.vertexCount      = h_mesh.vertexCount;
    h_gpuMesh.indexStart       = h_mesh.indexStart;
    h_gpuMesh.indexCount       = h_mesh.indexCount;
    
    h_gpuMesh.materialIndex    = h_mesh.materialIndex;

    h_gpuMesh.aabb = h_mesh.aabb;

    if(h_mesh.blas.nodeCount > 0)
        h_gpuMesh.blas = BVHToGPU(h_mesh.blas);

    Mesh_GPU* d_gpuMesh = nullptr;

    cudaError_t err;
    err = cudaMalloc((void**)&d_gpuMesh, sizeof(Mesh_GPU));
    err = cudaMemcpy(d_gpuMesh, &h_gpuMesh, sizeof(Mesh_GPU), cudaMemcpyHostToDevice);

    return d_gpuMesh;
}


#endif