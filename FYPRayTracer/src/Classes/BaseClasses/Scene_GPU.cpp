#include "Scene_GPU.h"

#include <iostream>

#include "../../Utility/CUDAUtils.h"

Scene_GPU SceneToGPU(const Scene& cpuScene)
{
    Scene_GPU gpuScene;
    
    // Copy geometry
    CopyVectorToDevice(cpuScene.vertices, gpuScene.vertices, gpuScene.vertexCount);
    CopyVectorToDevice(cpuScene.worldVertices, gpuScene.worldVertices, gpuScene.worldVertexCount);
    CopyVectorToDevice(cpuScene.triangleVertexIndices, gpuScene.triangleVertexIndices, gpuScene.triangleVertexIndexCount);
    CopyVectorToDevice(cpuScene.triangles, gpuScene.triangles, gpuScene.triangleCount);
    CopyVectorToDevice(cpuScene.meshes, gpuScene.meshes, gpuScene.meshCount);
    CopyVectorToDevice(cpuScene.materials, gpuScene.materials, gpuScene.materialCount);

    // Copy BVH
    BVHToGPU(cpuScene.bvh, gpuScene.bvh);
    
    return gpuScene;
}

void FreeSceneGPU(Scene_GPU& scene)
{
    if (scene.vertices) cudaFree(scene.vertices);
    if (scene.worldVertices) cudaFree(scene.worldVertices);
    if (scene.triangleVertexIndices) cudaFree(scene.triangleVertexIndices);
    if (scene.triangles) cudaFree(scene.triangles);
    if (scene.meshes) cudaFree(scene.meshes);
    if (scene.materials) cudaFree(scene.materials);

    FreeBVH_GPU(scene.bvh); // free BVH GPU memory
}
