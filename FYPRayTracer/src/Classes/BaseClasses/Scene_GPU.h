// Scene_GPU.h
#pragma once
#define GLM_FORCE_CUDA
#include <cuda_runtime.h>
#include "Scene.h" 

struct Scene_GPU
{
    Vertex* vertices;
    Vertex* worldVertices;
    uint32_t* triangleVertexIndices;
    Triangle* triangles;
    Mesh* meshes;
    Material* materials;

    BVH bvh; // CUDA-friendly BVH

    uint32_t vertexCount;
    uint32_t worldVertexCount;
    uint32_t triangleVertexIndexCount;
    uint32_t triangleCount;
    uint32_t meshCount;
    uint32_t materialCount;
};



// Convert CPU Scene to GPU Scene
Scene_GPU SceneToGPU(const Scene& cpuScene);

// Free GPU Scene memory
void FreeSceneGPU(Scene_GPU& scene);
