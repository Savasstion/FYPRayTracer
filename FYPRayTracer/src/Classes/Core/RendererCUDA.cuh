#ifndef RENDERER_CUDA_CUH
#define RENDERER_CUDA_CUH

#include <glm/glm.hpp>
#include "../BaseClasses/Ray.h"
#include "../BaseClasses/Scene.h"
#include "../BaseClasses/Camera.h"

struct Settings
{
    bool toAccumulate = true;
    int lightBounces = 1;
    int sampleCount = 1;
    glm::vec3 skyColor{1, 1, 1};
};

struct RendererGPU
{
    __host__ __device__ static glm::vec4 PerPixel(
        uint32_t x, uint32_t y,
        uint8_t maxBounces, uint8_t sampleCount,
        uint32_t frameIndex, const Settings& settings,
        const Scene* activeScene, const Camera* activeCamera,
        uint32_t imageWidth);

    __host__ __device__ static RayHitPayload TraceRay(const Ray& ray, const Scene* activeScene);
    __host__ __device__ static RayHitPayload ClosestHit(const Ray& ray, float hitDistance, int objectIndex, float u, float v, const Scene* activeScene);
    __host__ __device__ static RayHitPayload Miss(const Ray& ray);
    __host__ __device__ static glm::vec3 CalculateBRDF(
        const glm::vec3& N, const glm::vec3& V, const glm::vec3& L,
        const glm::vec3& albedo, float metallic, float roughness);
};

__global__ void RenderKernel(
    glm::vec4* accumulationData, uint32_t* renderImageData,
    uint32_t width, uint32_t height, uint32_t frameIndex,
    Settings settings, const Scene* scene, const Camera* camera);

#endif
