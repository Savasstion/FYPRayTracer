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

    // Copy TLAS
    gpuScene.tlas = BVHToGPU(cpuScene.tlas);

    // Copy filled Scene_GPU struct to device
    err = cudaMemcpy(d_scene, &gpuScene, sizeof(Scene_GPU), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy Scene_GPU error: " << cudaGetErrorString(err) << std::endl;
    }

//// Debug: Copy back first mesh BLAS from device
// if (!cpuScene.meshes.empty())
// {
//     // Step 1: Copy first mesh struct from device
//     Mesh_GPU debugMesh{};
//     err = cudaMemcpy(&debugMesh, gpuScene.meshes, sizeof(Mesh_GPU), cudaMemcpyDeviceToHost);
//     if (err != cudaSuccess) {
//         std::cerr << "cudaMemcpy (device->host) Mesh_GPU failed: " << cudaGetErrorString(err) << std::endl;
//     } else if (debugMesh.blas) {
//         // Step 2: Copy BVH struct from device
//         BVH h_bvh{};
//         err = cudaMemcpy(&h_bvh, debugMesh.blas, sizeof(BVH), cudaMemcpyDeviceToHost);
//         if (err != cudaSuccess) {
//             std::cerr << "cudaMemcpy (device->host) BVH failed: " << cudaGetErrorString(err) << std::endl;
//         } else {
//             std::cout << "Debug BVH: nodeCount=" << h_bvh.nodeCount
//                       << ", objectCount=" << h_bvh.objectCount
//                       << ", rootIndex=" << h_bvh.rootIndex << std::endl;
//
//             // Step 3: Copy nodes array if it exists
//             if (h_bvh.nodes && h_bvh.nodeCount > 0) {
//                 std::vector<BVH::Node> hostNodes(h_bvh.nodeCount);
//                 err = cudaMemcpy(hostNodes.data(), h_bvh.nodes,
//                                  h_bvh.nodeCount * sizeof(BVH::Node),
//                                  cudaMemcpyDeviceToHost);
//                 if (err != cudaSuccess) {
//                     std::cerr << "cudaMemcpy (device->host) BVH nodes failed: "
//                               << cudaGetErrorString(err) << std::endl;
//                 }
//                 auto n0 = hostNodes[0];
//                 auto n1 = hostNodes[1];
//                 auto n2 = hostNodes[2];
//                 auto n3 = hostNodes[3];
//                 auto n4 = hostNodes[4];
//                 auto n5 = hostNodes[5];
//                 auto n6 = hostNodes[6];
//                 
//             }
//         }
//     }
// }
    
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
        // Copy whole array of meshes back to host
        std::vector<Mesh_GPU> h_meshes(h_scene.meshCount);
        err = cudaMemcpy(h_meshes.data(), h_scene.meshes, 
                         h_scene.meshCount * sizeof(Mesh_GPU), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "cudaMemcpy error (meshes array): " << cudaGetErrorString(err) << std::endl;
        }

        for (size_t i = 0; i < h_scene.meshCount; i++)
        {
            if (h_meshes[i].blas)
                FreeBVH_GPU(h_meshes[i].blas);
        }

        // free mesh array itself
        cudaFree(h_scene.meshes);
    }

    // Free other arrays
    if (h_scene.vertices)              cudaFree(h_scene.vertices);
    if (h_scene.worldVertices)         cudaFree(h_scene.worldVertices);
    if (h_scene.triangleVertexIndices) cudaFree(h_scene.triangleVertexIndices);
    if (h_scene.triangles)             cudaFree(h_scene.triangles);
    if (h_scene.materials)             cudaFree(h_scene.materials);

    // Free TLAS
    if (h_scene.tlas) FreeBVH_GPU(h_scene.tlas);

    // Finally free the Scene_GPU struct itself
    cudaFree(d_scene);
}
