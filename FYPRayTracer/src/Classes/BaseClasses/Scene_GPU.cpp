#include "Scene_GPU.h"
#include <iostream>

#include "../../Utility/CUDAUtils.h"

Scene_GPU* SceneToGPU(const Scene& cpuScene)
{
    cudaError_t err;
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuda shit error: " << cudaGetErrorString(err) << std::endl;
    }
    
    Scene_GPU* d_scene = nullptr;
    err = cudaMalloc(&d_scene, sizeof(Scene_GPU));

    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Host-side Scene_GPU struct to fill in before copying to device
    Scene_GPU gpuScene{};
    gpuScene.vertices = nullptr;
    gpuScene.worldVertices = nullptr;
    gpuScene.triangleVertexIndices = nullptr;
    gpuScene.triangles = nullptr;
    gpuScene.meshes = nullptr;
    gpuScene.materials = nullptr;
    gpuScene.blasArray = nullptr;

    // Copy geometry arrays
    CopyVectorToDevice(cpuScene.vertices, gpuScene.vertices, gpuScene.vertexCount);
    CopyVectorToDevice(cpuScene.worldVertices, gpuScene.worldVertices, gpuScene.worldVertexCount);
    CopyVectorToDevice(cpuScene.triangleVertexIndices, gpuScene.triangleVertexIndices, gpuScene.triangleVertexIndexCount);
    CopyVectorToDevice(cpuScene.triangles, gpuScene.triangles, gpuScene.triangleCount);
    CopyVectorToDevice(cpuScene.meshes, gpuScene.meshes, gpuScene.meshCount);
    CopyVectorToDevice(cpuScene.materials, gpuScene.materials, gpuScene.materialCount);

    // Copy TLAS to device
    gpuScene.tlas = BVHToGPU(cpuScene.tlas);

    // Copy BLAS array
    gpuScene.blasCount = static_cast<uint32_t>(cpuScene.blasOfSceneMeshes.size());
    if (gpuScene.blasCount > 0)
    {
        std::vector<BVH*> blasPtrsHost(gpuScene.blasCount);

        for (uint32_t i = 0; i < gpuScene.blasCount; i++)
        {
            blasPtrsHost[i] = BVHToGPU(cpuScene.blasOfSceneMeshes[i]);
        }

        // Allocate array of BVH pointers on device
        err = cudaMalloc(&gpuScene.blasArray, gpuScene.blasCount * sizeof(BVH));
        if(err != cudaSuccess)
        {
            std::cerr << "cudaMalloc error: " << cudaGetErrorString(err) << std::endl;
        }
        
        err = cudaMemcpy(gpuScene.blasArray, blasPtrsHost.data(),
                   gpuScene.blasCount * sizeof(BVH),
                   cudaMemcpyHostToDevice);
        if(err != cudaSuccess)
        {
            std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
        }
    }
    
    // Copy filled struct from host to device
    err = cudaMemcpy(d_scene, &gpuScene, sizeof(Scene_GPU), cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
    }

    return d_scene;
}

void FreeSceneGPU(Scene_GPU* d_scene)
{
    cudaError_t err;
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuda shit error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy the device struct to host
    Scene_GPU h_scene;
    err = cudaMemcpy(&h_scene, d_scene, sizeof(Scene_GPU), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaFree(h_scene.vertices);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaFree(h_scene.worldVertices);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaFree(h_scene.triangleVertexIndices);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaFree(h_scene.triangles);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaFree(h_scene.meshes);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaFree(h_scene.materials);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }

    //FreeBVH_GPU(scene.bvh); // free BVH GPU memory

    err = cudaFree(d_scene);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }
}
