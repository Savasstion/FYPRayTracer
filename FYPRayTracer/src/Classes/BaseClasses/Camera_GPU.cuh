#ifndef CAMERA_GPU_H
#define CAMERA_GPU_H

#define GLM_FORCE_CUDA
#include <cuda.h>
#include "Camera.h"
#include <cuda_runtime.h>


struct Camera_GPU
{
    glm::mat4 projection;
    glm::mat4 view;
    glm::mat4 prevProjection;
    glm::mat4 prevView;
    glm::mat4 inverseProjection;
    glm::mat4 inverseView;

    float verticalFOV;
    float nearClip;
    float farClip;

    glm::vec3 position;
    glm::vec3 forwardDirection;

    glm::vec2 viewportSize; // width, height

    // Ray directions stored in a GPU-friendly way (flat array)
    glm::vec3* rayDirections; // device pointer

    __device__ __forceinline__ glm::vec3 GetRayDirection(uint32_t x, uint32_t y) const
    {
        uint32_t index = y * (uint32_t)viewportSize.x + x;
        return rayDirections[index];
    }

    __device__ __forceinline__ glm::vec2 GetNormalizedDeviceCoords(uint32_t x, uint32_t y) const
    {
        float W = viewportSize.x;
        float H = viewportSize.y;

        // Convert integer pixel coordinates → normalized [0..1]
        float u = (static_cast<float>(x) + 0.5f) / W;
        float v = (static_cast<float>(y) + 0.5f) / H;

        // Convert to OpenGL NDC [-1..1]
        float ndcX = u * 2.0f - 1.0f;
        float ndcY = v * 2.0f - 1.0f;

        // IMPORTANT: invert Y for OpenGL (screen Y grows upward)
        ndcY = -ndcY;

        return {ndcX, ndcY};
    }
};

Camera_GPU* CameraToGPU(Camera& cpuCam);
void FreeCameraGPU(Camera_GPU* d_camera);

#endif
