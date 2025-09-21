#ifndef RENDERER_CUDA_CUH
#define RENDERER_CUDA_CUH

#include "../BaseClasses/Ray.h"
#include "../BaseClasses/Scene_GPU.h"
#include "../BaseClasses/Camera_GPU.h"
#include "../BaseClasses/RenderingSettings.h"

struct RendererGPU
{
    __host__ __device__ static glm::vec4 PerPixel_BruteForce(
            uint32_t x, uint32_t y,
            uint8_t maxBounces, uint8_t sampleCount,
            uint32_t frameIndex, const RenderingSettings& settings,
            const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
            uint32_t imageWidth);

    __host__ __device__ static glm::vec4 PerPixel_UniformSampling(
            uint32_t x, uint32_t y,
            uint8_t maxBounces, uint8_t sampleCount,
            uint32_t frameIndex, const RenderingSettings& settings,
            const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
            uint32_t imageWidth);
    
    __host__ __device__ static glm::vec4 PerPixel_CosineWeightedSampling(
        uint32_t x, uint32_t y,
        uint8_t maxBounces, uint8_t sampleCount,
        uint32_t frameIndex, const RenderingSettings& settings,
        const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
        uint32_t imageWidth);

    __host__ __device__ static glm::vec4 PerPixel_GGXSampling(
        uint32_t x, uint32_t y,
        uint8_t maxBounces, uint8_t sampleCount,
        uint32_t frameIndex, const RenderingSettings& settings,
        const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
        uint32_t imageWidth);

    __host__ __device__ static glm::vec4 PerPixel_BRDFSampling(
        uint32_t x, uint32_t y,
        uint8_t maxBounces, uint8_t sampleCount,
        uint32_t frameIndex, const RenderingSettings& settings,
        const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
        uint32_t imageWidth);

    __host__ __device__ static glm::vec4 PerPixel_LightSourceSampling(
        uint32_t x, uint32_t y,
        uint8_t maxBounces, uint8_t sampleCount,
        uint32_t frameIndex, const RenderingSettings& settings,
        const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
        uint32_t imageWidth);

    __host__ __device__ static glm::vec4 PerPixel_NextEventEstimation(
        uint32_t x, uint32_t y,
        uint8_t maxBounces, uint8_t sampleCount,
        uint32_t frameIndex, const RenderingSettings& settings,
        const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
        uint32_t imageWidth);

    // __host__ __device__ static glm::vec4 RendererGPU::PerPixel_NextEventEstimation_MIS(
    //     uint32_t x, uint32_t y,
    //     uint8_t maxBounces, uint8_t sampleCount,
    //     uint32_t frameIndex, const RenderingSettings& settings,
    //     const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
    //     uint32_t imageWidth);

    __host__ __device__ static RayHitPayload TraceRay(
        const Ray& ray, const Scene_GPU* activeScene);

    __host__ __device__ static RayHitPayload ClosestHit(
        const Ray& ray, float hitDistance, int objectIndex,
        float u, float v, const Scene_GPU* activeScene);

    __host__ __device__ static RayHitPayload Miss(
        const Ray& ray);

    __host__ __device__ static glm::vec3 CalculateBRDF(
        const glm::vec3& N, const glm::vec3& V, const glm::vec3& L,
        const glm::vec3& albedo, float metallic, float roughness);

    static Scene_GPU* d_currentScene;
};

__global__ void RenderKernel(
    glm::vec4* accumulationData,
    uint32_t* renderImageData,
    uint32_t width,
    uint32_t height,
    uint32_t frameIndex,
    RenderingSettings settings,
    const Scene_GPU* scene,
    const Camera_GPU* camera
);

#endif
