#include "Scene_GPU.h"
#include <iostream>

#include "../../Utility/CUDAUtils.h"

Scene_GPU* SceneToGPU(const Scene& cpuScene)
{
    cudaError_t err;

    // Allocate device Scene_GPU struct
    Scene_GPU* d_scene = nullptr;
    err = cudaMalloc(&d_scene, sizeof(Scene_GPU));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc Scene_GPU error: " << cudaGetErrorString(err) << std::endl;
        return nullptr;
    }

    // Prepare host copy of Scene_GPU
    Scene_GPU gpuScene{};
    gpuScene.vertices = nullptr;
    gpuScene.worldVertices = nullptr;
    gpuScene.triangleVertexIndices = nullptr;
    gpuScene.triangles = nullptr;
    gpuScene.meshes = nullptr;
    gpuScene.materials = nullptr;
    gpuScene.tlas = nullptr;

    // Copy CPU vectors to GPU arrays
    CopyVectorToDevice(cpuScene.vertices, gpuScene.vertices, gpuScene.vertexCount);
    CopyVectorToDevice(cpuScene.worldVertices, gpuScene.worldVertices, gpuScene.worldVertexCount);
    CopyVectorToDevice(cpuScene.triangleVertexIndices, gpuScene.triangleVertexIndices, gpuScene.triangleVertexIndexCount);
    CopyVectorToDevice(cpuScene.triangles, gpuScene.triangles, gpuScene.triangleCount);
    CopyVectorToDevice(cpuScene.materials, gpuScene.materials, gpuScene.materialCount);

    // Copy TLAS
    gpuScene.tlas = BVHToGPU(cpuScene.tlas);

    // Copy filled Scene_GPU struct to device
    err = cudaMemcpy(d_scene, &gpuScene, sizeof(Scene_GPU), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy Scene_GPU error: " << cudaGetErrorString(err) << std::endl;
    }

    return d_scene;
}

void FreeSceneGPU(Scene_GPU* d_scene)
{
    if (!d_scene) return;
    cudaError_t err;

    // Copy device Scene_GPU struct to host
    Scene_GPU h_scene{};
    err = cudaMemcpy(&h_scene, d_scene, sizeof(Scene_GPU), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
    }

    // Free all arrays
    if (h_scene.vertices)                cudaFree(h_scene.vertices);
    if (h_scene.worldVertices)           cudaFree(h_scene.worldVertices);
    if (h_scene.triangleVertexIndices)   cudaFree(h_scene.triangleVertexIndices);
    if (h_scene.triangles)               cudaFree(h_scene.triangles);
    if (h_scene.meshes)                  cudaFree(h_scene.meshes);
    if (h_scene.materials)               cudaFree(h_scene.materials);

    // Free TLAS
    if (h_scene.tlas)                     FreeBVH_GPU(h_scene.tlas);

    // Free the Scene_GPU struct itself
    cudaFree(d_scene);
}
