#ifndef SCENE_GPU_H
#define SCENE_GPU_H
#include "Mesh_GPU.h"
#include "Scene.h" 

struct Scene_GPU
{
    Vertex* vertices;
    Vertex* worldVertices;
    uint32_t* triangleVertexIndices;
    Triangle* triangles;
    uint32_t* emissiveTriangles;
    Mesh_GPU* meshes;
    Material* materials;
    Texture* textures;

    BVH* tlas;
    LightTree* lightTree_tlas;

    uint32_t vertexCount;
    uint32_t worldVertexCount;
    uint32_t triangleVertexIndexCount;
    uint32_t triangleCount;
    uint32_t emissiveTriangleCount;
    uint32_t meshCount;
    uint32_t materialCount;
    uint32_t textureCount;
};



// Convert CPU Scene to GPU Scene
Scene_GPU* SceneToGPU(const Scene& cpuScene);

// Free GPU Scene memory
void FreeSceneGPU(Scene_GPU* d_scene);



#endif