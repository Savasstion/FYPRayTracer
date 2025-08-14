#ifndef CAMERA_GPU_H
#define CAMERA_GPU_H

#define GLM_FORCE_CUDA
#include "Camera.h"
#include <cuda_runtime.h>


struct Camera_GPU
{
    glm::mat4 projection;
    glm::mat4 view;
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

    __device__ glm::vec3 GetRayDirection(uint32_t x, uint32_t y) const
    {
        uint32_t index = y * (uint32_t)viewportSize.x + x;
        return rayDirections[index];
    }
};

Camera_GPU* CameraToGPU(const Camera& cpuCam);
void FreeCameraGPU(Camera_GPU* d_camera);

#endif
