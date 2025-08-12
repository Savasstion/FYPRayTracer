#include "Camera_GPU.h"

Camera_GPU CameraToGPU(const Camera& cpuCam)
{
    Camera_GPU gpuCam{};
    gpuCam.projection        = cpuCam.GetProjection();
    gpuCam.view              = cpuCam.GetView();
    gpuCam.inverseProjection = cpuCam.GetInverseProjection();
    gpuCam.inverseView       = cpuCam.GetInverseView();
    gpuCam.verticalFOV       = 45.0f; // Or cpuCam.GetVerticalFOV() if you add getter
    gpuCam.nearClip          = 0.1f;
    gpuCam.farClip           = 100.0f;
    gpuCam.position          = cpuCam.GetPosition();
    gpuCam.forwardDirection  = cpuCam.GetDirection();

    const auto& rayDirs = cpuCam.GetRayDirections();
    uint32_t width  = cpuCam.GetRayDirections().size() ? cpuCam.GetViewportWidth() : 0;
    uint32_t height = width ? cpuCam.GetViewportHeight() : 0;
    gpuCam.viewportSize = glm::vec2(width, height);

    // Allocate and copy ray directions to device
    size_t rayDirCount = rayDirs.size();
    cudaMalloc(&gpuCam.rayDirections, sizeof(glm::vec3) * rayDirCount);
    cudaMemcpy(gpuCam.rayDirections, rayDirs.data(),
               sizeof(glm::vec3) * rayDirCount, cudaMemcpyHostToDevice);

    return gpuCam;
}
