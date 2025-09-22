#include "Camera_GPU.h"
#include <iostream>

__host__ Camera_GPU* CameraToGPU(Camera& cpuCam)
{
    cudaError_t err;
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuda shit error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Create a host-side Camera_GPU struct to fill in
    Camera_GPU gpuCam{};
    gpuCam.projection        = cpuCam.GetProjection();
    gpuCam.view              = cpuCam.GetView();
    gpuCam.inverseProjection = cpuCam.GetInverseProjection();
    gpuCam.inverseView       = cpuCam.GetInverseView();
    gpuCam.verticalFOV       = 45.0f; // Or cpuCam.GetVerticalFOV() if you have a getter
    gpuCam.nearClip          = 0.1f;
    gpuCam.farClip           = 100.0f;
    gpuCam.position          = cpuCam.GetPosition();
    gpuCam.forwardDirection  = cpuCam.GetDirection();

    const auto& rayDirs = cpuCam.GetRayDirections();
    uint32_t width  = cpuCam.GetRayDirections().size() ? cpuCam.GetViewportWidth() : 0;
    uint32_t height = width ? cpuCam.GetViewportHeight() : 0;
    gpuCam.viewportSize = glm::vec2(width, height);

    // Allocate rayDirections array on device
    size_t rayDirCount = rayDirs.size();
    if (rayDirCount > 0)
    {
        glm::vec3* d_rayDirs = nullptr;
        cudaMalloc((void**)&d_rayDirs, sizeof(glm::vec3) * rayDirCount);
        cudaMemcpy(d_rayDirs, rayDirs.data(),
                   sizeof(glm::vec3) * rayDirCount,
                   cudaMemcpyHostToDevice);
        gpuCam.rayDirections = d_rayDirs;
    }
    else
    {
        gpuCam.rayDirections = nullptr;
    }

    // Allocate Camera_GPU struct on device
    Camera_GPU* d_gpuCam = nullptr;
    cudaMalloc((void**)&d_gpuCam, sizeof(Camera_GPU));

    // Copy filled struct from host to device
    err = cudaMemcpy(d_gpuCam, &gpuCam, sizeof(Camera_GPU), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
    }
    return d_gpuCam;
}

void FreeCameraGPU(Camera_GPU* d_camera)
{
    cudaError_t err;

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuda shit error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy the device struct to host
    Camera_GPU h_camera;
    err = cudaMemcpy(&h_camera, d_camera, sizeof(Camera_GPU), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaFree(h_camera.rayDirections);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }

    err = cudaFree(d_camera);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }
}
