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
    gpuScene.lightTree_tlas = nullptr;

    // Copy CPU vectors to GPU arrays
    CopyVectorToDevice(cpuScene.vertices, gpuScene.vertices, gpuScene.vertexCount);
    CopyVectorToDevice(cpuScene.worldVertices, gpuScene.worldVertices, gpuScene.worldVertexCount);
    CopyVectorToDevice(cpuScene.triangleVertexIndices, gpuScene.triangleVertexIndices, gpuScene.triangleVertexIndexCount);
    CopyVectorToDevice(cpuScene.triangles, gpuScene.triangles, gpuScene.triangleCount);
    CopyVectorToDevice(cpuScene.materials, gpuScene.materials, gpuScene.materialCount);

    //  Copy Meshes
    err = cudaMalloc((void**)&gpuScene.meshes, sizeof(Mesh_GPU) * cpuScene.meshes.size());
    if(err != cudaSuccess)
        std::cerr << "cudaMalloc failed\n";
    
    for(size_t i = 0; i < cpuScene.meshes.size(); i++)
    {
        Mesh_GPU h_gpuMesh = MeshToHostMeshGPU(cpuScene.meshes[i]);
        err = cudaMemcpy(&gpuScene.meshes[i], &h_gpuMesh, sizeof(Mesh_GPU), cudaMemcpyHostToDevice);
        if(err != cudaSuccess)
            std::cerr << "cudaMemcpy failed\n";
    }
    gpuScene.meshCount = static_cast<uint32_t>(cpuScene.meshes.size());

    //  Copy Textures
    err = cudaMalloc((void**)&gpuScene.textures, sizeof(Texture) * cpuScene.textures.size());
    if(err != cudaSuccess)
        std::cerr << "cudaMalloc failed\n";

    std::vector<Texture> texBuffer;
    for(size_t i = 0; i < cpuScene.textures.size(); i++)
        texBuffer.push_back(TextureToHostTextureGPU(cpuScene.textures[i]));
    
    err = cudaMemcpy(gpuScene.textures, texBuffer.data(), sizeof(Texture), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        std::cerr << "cudaMemcpy failed\n";
    gpuScene.textureCount = static_cast<uint32_t>(cpuScene.textures.size());

    // Copy TLAS
    gpuScene.tlas = BVHToGPU(cpuScene.tlas);
    // Copy Light Tree
    gpuScene.lightTree_tlas = LightTreeToGPU(cpuScene.lightTree_tlas);

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

    // Copy device Scene_GPU struct to host
    Scene_GPU h_scene{};
    cudaError_t err = cudaMemcpy(&h_scene, d_scene, sizeof(Scene_GPU), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // Free all meshes
    if (h_scene.meshes)
    {
        // Copy mesh array back to host so we can inspect blas pointers
        std::vector<Mesh_GPU> hostMeshes(h_scene.meshCount);
        err = cudaMemcpy(hostMeshes.data(), h_scene.meshes,
            sizeof(Mesh_GPU) * h_scene.meshCount,
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy meshes error: " << cudaGetErrorString(err) << std::endl;
        }

        // Free each BLAS
        for (size_t i = 0; i < h_scene.meshCount; i++) {
            if (hostMeshes[i].blas) {
                FreeBVH_GPU(hostMeshes[i].blas);
            }
        }

        // Now free the device mesh array itself
        err = cudaFree(h_scene.meshes);
        if (err != cudaSuccess) {
            std::cerr << "cudaFree mesh array error: " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Free other arrays
    if (h_scene.vertices)              cudaFree(h_scene.vertices);
    if (h_scene.worldVertices)         cudaFree(h_scene.worldVertices);
    if (h_scene.triangleVertexIndices) cudaFree(h_scene.triangleVertexIndices);
    if (h_scene.triangles)             cudaFree(h_scene.triangles);
    if (h_scene.materials)             cudaFree(h_scene.materials);

    // Free TLAS
    if (h_scene.tlas) FreeBVH_GPU(h_scene.tlas);
    // Free Light Tree
    if (h_scene.lightTree_tlas) FreeLightTree_GPU(h_scene.lightTree_tlas);

    // Finally free the Scene_GPU struct itself
    cudaFree(d_scene);
}
