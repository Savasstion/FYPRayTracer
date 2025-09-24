#include "Renderer.h"
#include "RendererCUDA.cuh"
#include "../../Utility/ColorUtils.cuh"
#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <iostream>
#include <glm/gtx/compatibility.hpp>

Scene_GPU* RendererGPU::d_currentScene = nullptr;

void Renderer::Render(Scene& scene, Camera& camera)
{
    m_ActiveScene = &scene;
    m_ActiveCamera = &camera;
    
    uint32_t width  = m_FinalRenderImage->GetWidth();
    uint32_t height = m_FinalRenderImage->GetHeight();
    size_t pixelCount = width * height;
    
    constexpr size_t vec4Size = sizeof(glm::vec4);
    constexpr size_t uint32Size = sizeof(uint32_t);

    // Allocate device buffers for accumulation and output image
    glm::vec4* d_accumulationData = nullptr;
    uint32_t* d_renderImageData = nullptr;

    cudaError_t err;
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuda shit error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaMalloc((void**)&d_accumulationData, pixelCount * vec4Size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaMalloc((void**)&d_renderImageData, pixelCount * uint32Size);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc error: " << cudaGetErrorString(err) << std::endl;
    }

    // Initialize or copy accumulation buffer
    if (m_FrameIndex == 1)
        err = cudaMemset(d_accumulationData, 0, pixelCount * vec4Size);
    else
        err = cudaMemcpy(d_accumulationData, m_AccumulationData, pixelCount * vec4Size, cudaMemcpyHostToDevice);
    
    if (err != cudaSuccess) {
        std::cerr << "cuda copy error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Allocate device versions
    Scene_GPU* d_sceneGPU = RendererGPU::d_currentScene;
    if(isSceneUpdated)
    {
        if(d_sceneGPU != nullptr)
            FreeSceneGPU(d_sceneGPU);
        d_sceneGPU = SceneToGPU(scene);
        RendererGPU::d_currentScene = d_sceneGPU;
        isSceneUpdated = false;
    }
    Camera_GPU* d_cameraGPU = CameraToGPU(camera);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuda shit error: " << cudaGetErrorString(err) << std::endl;
    }
    
    
    // Configure kernel launch
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    //  Do per pixel rendering task in parallel for all pixels
    RenderKernel<<<numBlocks, threadsPerBlock>>>(
        d_accumulationData,
        d_renderImageData,
        width,
        height,
        m_FrameIndex,
        m_Settings,
        d_sceneGPU,
        d_cameraGPU
    );
    
     err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Copy results back
    err = cudaMemcpy(m_AccumulationData, d_accumulationData, pixelCount * vec4Size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaMemcpy(m_RenderImageData, d_renderImageData, pixelCount * uint32Size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy error: " << cudaGetErrorString(err) << std::endl;
    }
    
    m_FinalRenderImage->SetData(m_RenderImageData);

    if (m_Settings.toAccumulate)
        m_FrameIndex++;
    else
        m_FrameIndex = 1;


    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "cuda shit error: " << cudaGetErrorString(err) << std::endl;
    }
    
    // Free device memory
    err = cudaFree(d_accumulationData);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }
    
    err = cudaFree(d_renderImageData);
    if (err != cudaSuccess) {
        std::cerr << "cudaFree error: " << cudaGetErrorString(err) << std::endl;
    }
    
    FreeCameraGPU(d_cameraGPU);
    
}


__host__ __device__ RayHitPayload RendererGPU::TraceRay(const Ray& ray, const Scene_GPU* activeScene)
{
    if (activeScene->triangleCount == 0) return Miss(ray);

    float closestHitDistance = FLT_MAX;
    int closestTriangle = -1;
    float closestU = 0.0f;
    float closestV = 0.0f;

    //TLAS - BLAS Traversal
     BVH* tlas = activeScene->tlas;
    
     const int TLAS_STACK_SIZE = 256;
     int tlasStack[TLAS_STACK_SIZE];
     int tlasStackTop = 0;
    
     const int BLAS_STACK_SIZE = 1024;
     int blasStack[BLAS_STACK_SIZE];
    
     tlasStack[tlasStackTop++] = static_cast<int>(tlas->rootIndex);
    
     while (tlasStackTop > 0)
     {
         int nodeIndex = tlasStack[--tlasStackTop];
    
         const BVH::Node& node = tlas->nodes[nodeIndex];
         if (!IntersectRayAABB(ray, node.box)) continue;
    
         if (node.isLeaf)
         {
             size_t meshIndex = node.objectIndex;
             
             BVH* blas = activeScene->meshes[meshIndex].blas;
             
             int blasStackTop = 0;
             blasStack[blasStackTop++] = static_cast<int>(blas->rootIndex);
             
             while (blasStackTop > 0)
             {
                 int bnodeIndex = blasStack[--blasStackTop];
    
                 const BVH::Node& bnode = blas->nodes[bnodeIndex];
                 if (!IntersectRayAABB(ray, bnode.box)) continue;
    
                 if (bnode.isLeaf)
                 {
                     size_t triangleIndex = bnode.objectIndex;

                     const Triangle& tri = activeScene->triangles[triangleIndex];
                     const glm::vec3& v0 = activeScene->worldVertices[tri.v0].position;
                     const glm::vec3& v1 = activeScene->worldVertices[tri.v1].position;
                     const glm::vec3& v2 = activeScene->worldVertices[tri.v2].position;

                     // Möller–Trumbore intersection algorithm
                     glm::vec3 edge1 = v1 - v0;
                     glm::vec3 edge2 = v2 - v0;
                     glm::vec3 h = glm::cross(ray.direction, edge2);
                     float a = glm::dot(edge1, h);
                     // if (fabsf(a) < 1e-8f) continue;
    
                     float f = 1.0f / a;
                     glm::vec3 s = ray.origin - v0;
                     float u = f * glm::dot(s, h);
                     if (u < 0.0f || u > 1.0f) continue;
    
                     glm::vec3 q = glm::cross(s, edge1);
                     float v = f * glm::dot(ray.direction, q);
                     if (v < 0.0f || (u + v) > 1.0f) continue;
    
                     float t = f * glm::dot(edge2, q);
    
                     if (t > 0.0001f && t < closestHitDistance)
                     {
                         closestHitDistance = t;
                         closestTriangle = static_cast<int>(triangleIndex);
                         closestU = u;
                         closestV = v;
                     }
                 }
                 else
                 {
                     if (bnode.child1 != static_cast<size_t>(-1) && blasStackTop < BLAS_STACK_SIZE)
                         blasStack[blasStackTop++] = static_cast<int>(bnode.child1);
                     if (bnode.child2 != static_cast<size_t>(-1) && blasStackTop < BLAS_STACK_SIZE)
                         blasStack[blasStackTop++] = static_cast<int>(bnode.child2);
                 }
             }
         }
         else
         {
             if (node.child1 != static_cast<size_t>(-1) && tlasStackTop < TLAS_STACK_SIZE)
                 tlasStack[tlasStackTop++] = static_cast<int>(node.child1);
             if (node.child2 != static_cast<size_t>(-1) && tlasStackTop < TLAS_STACK_SIZE)
                 tlasStack[tlasStackTop++] = static_cast<int>(node.child2);
         }
     }

    if (closestTriangle < 0)
        return Miss(ray);

    return ClosestHit(ray, closestHitDistance, closestTriangle, closestU, closestV, activeScene);
}


 //  PURE BRUTE-FORCE
__host__ __device__ glm::vec4 RendererGPU::PerPixel_BruteForce(
    uint32_t x, uint32_t y,
    uint8_t maxBounces, uint8_t sampleCount,
    uint32_t frameIndex, const RenderingSettings& settings,
    const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
    uint32_t imageWidth)
{
    //  PURE BRUTE-FORCE, EVERY FRAME ONLY CHOOSE ONE RANDOM PATH TO FOLLOW
    //  LIKE A OFFLINE-RENDERER, KEEPS TRACK OF SUM AND THEN AVERAGE THE PIXEL'S COLOR OVER TIME TO EVENTUALLY FORM A PHYSICALLY-ACCURATE IMAGE (keyword : "eventually")
    
    //  all directions within a hemisphere are all equally likely to be sampled
    
    uint32_t seed = x + y * imageWidth;
    seed *= frameIndex;

    glm::vec3 radiance{0.0f};   // Final color accumulated from all samples

    // == PRIMARY RAY ==
    Ray primaryRay;
    primaryRay.origin = activeCamera->position;
    primaryRay.direction = activeCamera->rayDirections[x + y * imageWidth];

    RayHitPayload primaryPayload = TraceRay(primaryRay, activeScene);

    // Hit sky immediately
    if (primaryPayload.hitDistance < 0.0f)
        return glm::vec4(settings.skyColor, 1.0f);

    const Triangle& hitTri = activeScene->triangles[primaryPayload.objectIndex];
    const Material& hitMaterial = activeScene->materials[hitTri.materialIndex];

    // Hit emissive object immediately
    if (glm::length(hitMaterial.GetEmission()) > 0.0f)
        return glm::vec4(hitMaterial.GetEmission(), 1.0f);
    
    glm::vec3 sampleThroughput{1.0f};
    Ray sampleRay;
    RayHitPayload samplePayload = primaryPayload;

    // Sample initial direction from first hit
    glm::vec3 newDir = MathUtils::UniformSampleHemisphere(primaryPayload.worldNormal, seed);

    glm::vec3 brdf = MathUtils::CalculateBRDF(
        primaryPayload.worldNormal,
        -primaryRay.direction,
        newDir,
        hitMaterial.albedo,
        hitMaterial.metallic,
        hitMaterial.roughness
    );

    float cosTheta = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);  //  Geometry Term
    float pdf = MathUtils::UniformHemispherePDF();
    sampleThroughput *= brdf * cosTheta / pdf; // Rendering equation core

    sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-12f;
    sampleRay.direction = newDir;

    // Trace the path for maxBounces
    for (int bounce = 0; bounce < maxBounces; bounce++)
    {
        seed += 31 * bounce;

        samplePayload = TraceRay(sampleRay, activeScene);

        // Hit sky
        if (samplePayload.hitDistance < 0.0f)
        {
            radiance += sampleThroughput * settings.skyColor;
            break;
        }

        const Triangle& tri = activeScene->triangles[samplePayload.objectIndex];
        const Material& material = activeScene->materials[tri.materialIndex];

        // Hit emissive light
        glm::vec3 emission = material.GetEmission();
        if (glm::length(emission) > 0.0f)
        {
            radiance += sampleThroughput * emission;
            break;
        }

        // Sample next direction
        glm::vec3 bounceDir = MathUtils::UniformSampleHemisphere(samplePayload.worldNormal, seed);
            
        glm::vec3 bounceBrdf = MathUtils::CalculateBRDF(
        samplePayload.worldNormal,
        -sampleRay.direction,
        bounceDir,
        material.albedo,
        material.metallic,
        material.roughness
        );

        float bounceCosTheta = glm::max(glm::dot(bounceDir, samplePayload.worldNormal), 0.0f);
        float bouncePdf = MathUtils::UniformHemispherePDF();
        sampleThroughput *= bounceBrdf * bounceCosTheta / bouncePdf;

        sampleRay.origin = samplePayload.worldPosition + samplePayload.worldNormal * 1e-12f;
        sampleRay.direction = bounceDir;
    }
    
    return glm::vec4(radiance, 1.0f);
}

 //  UNIFORM SAMPLING
__host__ __device__ glm::vec4 RendererGPU::PerPixel_UniformSampling(
    uint32_t x, uint32_t y,
    uint8_t maxBounces, uint8_t sampleCount,
    uint32_t frameIndex, const RenderingSettings& settings,
    const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
    uint32_t imageWidth)
{
    //  all directions within a hemisphere are all equally likely to be sampled
    
    uint32_t seed = x + y * imageWidth;
    seed *= frameIndex;

    glm::vec3 radiance{0.0f};   // Final color accumulated from all samples

    // == PRIMARY RAY ==
    Ray primaryRay;
    primaryRay.origin = activeCamera->position;
    primaryRay.direction = activeCamera->rayDirections[x + y * imageWidth];

    RayHitPayload primaryPayload = TraceRay(primaryRay, activeScene);

    // Hit sky immediately
    if (primaryPayload.hitDistance < 0.0f)
        return glm::vec4(settings.skyColor, 1.0f);

    const Triangle& hitTri = activeScene->triangles[primaryPayload.objectIndex];
    const Material& hitMaterial = activeScene->materials[hitTri.materialIndex];

    // Hit emissive object immediately
    if (glm::length(hitMaterial.GetEmission()) > 0.0f)
        return glm::vec4(hitMaterial.GetEmission(), 1.0f);

    // == SAMPLE MULTIPLE LIGHT PATHS FROM FIRST HIT ==
    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        seed += (sampleIndex + 1) * 27;
        glm::vec3 sampleThroughput{1.0f};
        Ray sampleRay;
        RayHitPayload samplePayload = primaryPayload;

        // Sample initial direction from first hit
        glm::vec3 newDir = MathUtils::UniformSampleHemisphere(primaryPayload.worldNormal, seed);

        glm::vec3 brdf = MathUtils::CalculateBRDF(
            primaryPayload.worldNormal,
            -primaryRay.direction,
            newDir,
            hitMaterial.albedo,
            hitMaterial.metallic,
            hitMaterial.roughness
        );

        float cosTheta = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);  //  Geometry Term
        float pdf = MathUtils::UniformHemispherePDF();
        sampleThroughput *= brdf * cosTheta / pdf; // Rendering equation core

        sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-12f;
        sampleRay.direction = newDir;

        // Trace the path for maxBounces
        for (int bounce = 0; bounce < maxBounces; bounce++)
        {
            seed += sampleIndex + 31 * bounce;

            samplePayload = TraceRay(sampleRay, activeScene);

            // Hit sky
            if (samplePayload.hitDistance < 0.0f)
            {
                radiance += sampleThroughput * settings.skyColor;
                break;
            }

            const Triangle& tri = activeScene->triangles[samplePayload.objectIndex];
            const Material& material = activeScene->materials[tri.materialIndex];

            // Hit emissive light
            glm::vec3 emission = material.GetEmission();
            if (glm::length(emission) > 0.0f)
            {
                radiance += sampleThroughput * emission;
                break;
            }

            // Sample next direction
            glm::vec3 bounceDir = MathUtils::UniformSampleHemisphere(samplePayload.worldNormal, seed);
            
            glm::vec3 bounceBrdf = MathUtils::CalculateBRDF(
                samplePayload.worldNormal,
                -sampleRay.direction,
                bounceDir,
                material.albedo,
                material.metallic,
                material.roughness
            );

            float bounceCosTheta = glm::max(glm::dot(bounceDir, samplePayload.worldNormal), 0.0f);
            float bouncePdf = MathUtils::UniformHemispherePDF();
            sampleThroughput *= bounceBrdf * bounceCosTheta / bouncePdf;

            sampleRay.origin = samplePayload.worldPosition + samplePayload.worldNormal * 1e-12f;
            sampleRay.direction = bounceDir;
        }
    }

    radiance /= float(sampleCount); // Average across all sampled paths
    return glm::vec4(radiance, 1.0f);
}

 //  COSINE-WEIGHTED SAMPLING
__host__ __device__ glm::vec4 RendererGPU::PerPixel_CosineWeightedSampling(
    uint32_t x, uint32_t y,
    uint8_t maxBounces, uint8_t sampleCount,
    uint32_t frameIndex, const RenderingSettings& settings,
    const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
    uint32_t imageWidth)
{
    //  a kind of BRDF sampling method that makes it so that shallower angles are less likely to be sampled as they likely contribute less light
    
    uint32_t seed = x + y * imageWidth;
    seed *= frameIndex;

    glm::vec3 radiance{0.0f};   // Final color accumulated from all samples

    // == PRIMARY RAY ==
    Ray primaryRay;
    primaryRay.origin = activeCamera->position;
    primaryRay.direction = activeCamera->rayDirections[x + y * imageWidth];

    RayHitPayload primaryPayload = TraceRay(primaryRay, activeScene);

    // Hit sky immediately
    if (primaryPayload.hitDistance < 0.0f)
        return glm::vec4(settings.skyColor, 1.0f);

    const Triangle& hitTri = activeScene->triangles[primaryPayload.objectIndex];
    const Material& hitMaterial = activeScene->materials[hitTri.materialIndex];

    // Hit emissive object immediately
    if (glm::length(hitMaterial.GetEmission()) > 0.0f)
        return glm::vec4(hitMaterial.GetEmission(), 1.0f);

    // == SAMPLE MULTIPLE LIGHT PATHS FROM FIRST HIT ==
    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        seed += (sampleIndex + 1) * 27;
        glm::vec3 sampleThroughput{1.0f};
        Ray sampleRay;
        RayHitPayload samplePayload = primaryPayload;

        // Sample initial direction from first hit
        glm::vec3 newDir = MathUtils::CosineSampleHemisphere(primaryPayload.worldNormal, seed);

        glm::vec3 brdf = MathUtils::CalculateBRDF(
            primaryPayload.worldNormal,
            -primaryRay.direction,
            newDir,
            hitMaterial.albedo,
            hitMaterial.metallic,
            hitMaterial.roughness
        );

        float cosTheta = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);  //  Geometry Term
        float pdf = MathUtils::CosineHemispherePDF(cosTheta);
        sampleThroughput *= brdf * cosTheta / pdf; // Rendering equation core

        sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-12f;
        sampleRay.direction = newDir;

        // Trace the path for maxBounces
        for (int bounce = 0; bounce < maxBounces; bounce++)
        {
            seed += sampleIndex + 31 * bounce;

            samplePayload = TraceRay(sampleRay, activeScene);

            // Hit sky
            if (samplePayload.hitDistance < 0.0f)
            {
                radiance += sampleThroughput * settings.skyColor;
                break;
            }

            const Triangle& tri = activeScene->triangles[samplePayload.objectIndex];
            const Material& material = activeScene->materials[tri.materialIndex];

            // Hit emissive light
            glm::vec3 emission = material.GetEmission();
            if (glm::length(emission) > 0.0f)
            {
                radiance += sampleThroughput * emission;
                break;
            }

            // Sample next direction
            glm::vec3 bounceDir = MathUtils::CosineSampleHemisphere(samplePayload.worldNormal, seed);
            
            glm::vec3 bounceBrdf = MathUtils::CalculateBRDF(
                samplePayload.worldNormal,
                -sampleRay.direction,
                bounceDir,
                material.albedo,
                material.metallic,
                material.roughness
            );

            float bounceCosTheta = glm::max(glm::dot(bounceDir, samplePayload.worldNormal), 0.0f);
            float bouncePdf = MathUtils::CosineHemispherePDF(bounceCosTheta);
            sampleThroughput *= bounceBrdf * bounceCosTheta / bouncePdf;

            sampleRay.origin = samplePayload.worldPosition + samplePayload.worldNormal * 1e-12f;
            sampleRay.direction = bounceDir;
        }
    }

    radiance /= float(sampleCount); // Average across all sampled paths
    return glm::vec4(radiance, 1.0f);
}

 //  GGX SAMPLING
__host__ __device__ glm::vec4 RendererGPU::PerPixel_GGXSampling(
    uint32_t x, uint32_t y,
    uint8_t maxBounces, uint8_t sampleCount,
    uint32_t frameIndex, const RenderingSettings& settings,
    const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
    uint32_t imageWidth)
{
    //  sample more often towards directions that contribute more to specular lighting
    
    uint32_t seed = x + y * imageWidth;
    seed *= frameIndex;

    glm::vec3 radiance{0.0f};   // Final color accumulated from all samples

    // == PRIMARY RAY ==
    Ray primaryRay;
    primaryRay.origin = activeCamera->position;
    primaryRay.direction = activeCamera->rayDirections[x + y * imageWidth];

    RayHitPayload primaryPayload = TraceRay(primaryRay, activeScene);

    // Hit sky immediately
    if (primaryPayload.hitDistance < 0.0f)
        return glm::vec4(settings.skyColor, 1.0f);

    const Triangle& hitTri = activeScene->triangles[primaryPayload.objectIndex];
    const Material& hitMaterial = activeScene->materials[hitTri.materialIndex];

    // Hit emissive object immediately
    if (glm::length(hitMaterial.GetEmission()) > 0.0f)
        return glm::vec4(hitMaterial.GetEmission(), 1.0f);

    // == SAMPLE MULTIPLE LIGHT PATHS FROM FIRST HIT ==
    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        seed += (sampleIndex + 1) * 27;
        glm::vec3 sampleThroughput{1.0f};
        Ray sampleRay;
        RayHitPayload samplePayload = primaryPayload;

        // Sample initial direction from first hit
        float pdf;
        glm::vec3 newDir = MathUtils::GGXSampleHemisphere(primaryPayload.worldNormal, -primaryRay.direction, hitMaterial.roughness,seed, pdf);

        glm::vec3 brdf = MathUtils::CalculateBRDF(
            primaryPayload.worldNormal,
            -primaryRay.direction,
            newDir,
            hitMaterial.albedo,
            hitMaterial.metallic,
            hitMaterial.roughness
        );

        float cosTheta = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);  //  Geometry Term
        sampleThroughput *= brdf * cosTheta / pdf; // Rendering equation core

        sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-12f;
        sampleRay.direction = newDir;

        // Trace the path for maxBounces
        for (int bounce = 0; bounce < maxBounces; bounce++)
        {
            seed += sampleIndex + 31 * bounce;

            samplePayload = TraceRay(sampleRay, activeScene);

            // Hit sky
            if (samplePayload.hitDistance < 0.0f)
            {
                radiance += sampleThroughput * settings.skyColor;
                break;
            }

            const Triangle& tri = activeScene->triangles[samplePayload.objectIndex];
            const Material& material = activeScene->materials[tri.materialIndex];

            // Hit emissive light
            glm::vec3 emission = material.GetEmission();
            if (glm::length(emission) > 0.0f)
            {
                radiance += sampleThroughput * emission;
                break;
            }

            // Sample next direction
            float bouncePdf;
            glm::vec3 bounceDir = MathUtils::GGXSampleHemisphere(samplePayload.worldNormal, -sampleRay.direction, hitMaterial.roughness,seed, bouncePdf);
            
            glm::vec3 bounceBrdf = MathUtils::CalculateBRDF(
                samplePayload.worldNormal,
                -sampleRay.direction,
                bounceDir,
                material.albedo,
                material.metallic,
                material.roughness
            );

            float bounceCosTheta = glm::max(glm::dot(bounceDir, samplePayload.worldNormal), 0.0f);
            sampleThroughput *= bounceBrdf * bounceCosTheta / bouncePdf;

            sampleRay.origin = samplePayload.worldPosition + samplePayload.worldNormal * 1e-12f;
            sampleRay.direction = bounceDir;
        }
    }

    radiance /= float(sampleCount); // Average across all sampled paths
    return glm::vec4(radiance, 1.0f);
}

//  BRDF SAMPLING
__host__ __device__ glm::vec4 RendererGPU::PerPixel_BRDFSampling(
    uint32_t x, uint32_t y,
    uint8_t maxBounces, uint8_t sampleCount,
    uint32_t frameIndex, const RenderingSettings& settings,
    const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
    uint32_t imageWidth)
{
    uint32_t seed = x + y * imageWidth;
    seed *= frameIndex;

    glm::vec3 radiance{0.0f};

    // PRIMARY RAY
    Ray primaryRay;
    primaryRay.origin = activeCamera->position;
    primaryRay.direction = activeCamera->rayDirections[x + y * imageWidth];

    RayHitPayload primaryPayload = TraceRay(primaryRay, activeScene);

    // Miss: hit sky
    if (primaryPayload.hitDistance < 0.0f)
        return glm::vec4(settings.skyColor, 1.0f);

    const Triangle& hitTri = activeScene->triangles[primaryPayload.objectIndex];
    const Material& hitMaterial = activeScene->materials[hitTri.materialIndex];

    // Hit emissive surface
    if (glm::length(hitMaterial.GetEmission()) > 0.0f)
        return glm::vec4(hitMaterial.GetEmission(), 1.0f);

    // MULTI-SAMPLE LOOP
    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        seed += (sampleIndex + 1) * 27;
        glm::vec3 sampleThroughput{1.0f};
        Ray sampleRay;
        RayHitPayload samplePayload = primaryPayload;

        // Sample initial bounce
        float pdf;
        glm::vec3 newDir = MathUtils::BRDFSampleHemisphere(
            primaryPayload.worldNormal,
            -primaryRay.direction,
            hitMaterial.albedo,
            hitMaterial.metallic,
            hitMaterial.roughness,
            seed,
            pdf
        );

        glm::vec3 brdf = MathUtils::CalculateBRDF(
            primaryPayload.worldNormal,
            -primaryRay.direction,
            newDir,
            hitMaterial.albedo,
            hitMaterial.metallic,
            hitMaterial.roughness
        );

        float cosTheta = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);
        sampleThroughput *= brdf * cosTheta / pdf;

        sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-12f;
        sampleRay.direction = newDir;

        // BOUNCE LOOP
        for (int bounce = 0; bounce < maxBounces; bounce++)
        {
            seed += sampleIndex + 31 * bounce;
            samplePayload = TraceRay(sampleRay, activeScene);

            // Miss: hit sky
            if (samplePayload.hitDistance < 0.0f)
            {
                radiance += sampleThroughput * settings.skyColor;
                break;
            }

            const Triangle& tri = activeScene->triangles[samplePayload.objectIndex];
            const Material& material = activeScene->materials[tri.materialIndex];

            // Hit emissive
            glm::vec3 emission = material.GetEmission();
            if (glm::length(emission) > 0.0f)
            {
                radiance += sampleThroughput * emission;
                break;
            }

            // Next bounce
            float bouncePdf;
            glm::vec3 bounceDir = MathUtils::BRDFSampleHemisphere(
                samplePayload.worldNormal,
                -sampleRay.direction,
                material.albedo,
                material.metallic,
                material.roughness,
                seed,
                bouncePdf
            );

            glm::vec3 bounceBrdf = MathUtils::CalculateBRDF(
                samplePayload.worldNormal,
                -sampleRay.direction,
                bounceDir,
                material.albedo,
                material.metallic,
                material.roughness
            );

            float bounceCosTheta = glm::max(glm::dot(bounceDir, samplePayload.worldNormal), 0.0f);
            sampleThroughput *= bounceBrdf * bounceCosTheta / bouncePdf;

            sampleRay.origin = samplePayload.worldPosition + samplePayload.worldNormal * 1e-12f;
            sampleRay.direction = bounceDir;
        }
    }

    radiance /= float(sampleCount);
    return glm::vec4(radiance, 1.0f);
}

//  LIGHT SOURCE SAMPLING
__host__ __device__ glm::vec4 RendererGPU::PerPixel_LightSourceSampling(
    uint32_t x, uint32_t y,
    uint8_t maxBounces, uint8_t sampleCount,
    uint32_t frameIndex, const RenderingSettings& settings,
    const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
    uint32_t imageWidth){
    uint32_t seed = x + y * imageWidth;
    seed *= frameIndex;

    glm::vec3 radiance{0.0f};

    // PRIMARY RAY
    Ray primaryRay;
    primaryRay.origin = activeCamera->position;
    primaryRay.direction = activeCamera->rayDirections[x + y * imageWidth];

    RayHitPayload primaryPayload = TraceRay(primaryRay, activeScene);

    // Miss: hit sky
    if (primaryPayload.hitDistance < 0.0f)
        return glm::vec4(settings.skyColor, 1.0f);

    const Triangle& hitTri = activeScene->triangles[primaryPayload.objectIndex];
    const Material& hitMaterial = activeScene->materials[hitTri.materialIndex];

    // Hit emissive surface
    if (glm::length(hitMaterial.GetEmission()) > 0.0f)
        return glm::vec4(hitMaterial.GetEmission(), 1.0f);

    // MULTI-SAMPLE LOOP
    for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
    {
        seed += (sampleIndex + 1) * 27;
        glm::vec3 sampleThroughput{1.0f};
        Ray sampleRay;
        RayHitPayload samplePayload = primaryPayload;

        // sample direction to light source
        LightTree::ShadingPointQuery sp;
        sp.normal = primaryPayload.worldNormal;
        sp.position = primaryPayload.worldPosition;
        LightTree::SampledLight sampledLight = PickLight_TLAS(activeScene->meshes, activeScene->lightTree_tlas, sp, seed);

        //  get emmisive triangle data
        glm::vec3 p0 = activeScene->worldVertices[activeScene->triangles[sampledLight.emitterIndex].v0].position;
        glm::vec3 p1 = activeScene->worldVertices[activeScene->triangles[sampledLight.emitterIndex].v1].position;
        glm::vec3 p2 = activeScene->worldVertices[activeScene->triangles[sampledLight.emitterIndex].v2].position;
        glm::vec3 n0 = activeScene->worldVertices[activeScene->triangles[sampledLight.emitterIndex].v0].normal;
        glm::vec3 n1 = activeScene->worldVertices[activeScene->triangles[sampledLight.emitterIndex].v1].normal;
        glm::vec3 n2 = activeScene->worldVertices[activeScene->triangles[sampledLight.emitterIndex].v2].normal;

        //  get new ray direction towards selected light source
        glm::vec3 emmisivePoint = Triangle::GetRandomPointOnTriangle(p0, p1, p2, seed);
        glm::vec3 newDir = emmisivePoint - primaryPayload.worldPosition;
        float distance = glm::distance(emmisivePoint, primaryPayload.worldPosition);
        newDir = newDir / distance;
        
        glm::vec3 brdf = MathUtils::CalculateBRDF(
            primaryPayload.worldNormal,
            -primaryRay.direction,
            newDir,
            hitMaterial.albedo,
            hitMaterial.metallic,
            hitMaterial.roughness
        );

        //  rendering equation
        float cosTheta_x = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);
        float cosTheta_y = glm::max(glm::dot(-newDir, Triangle::GetTriangleNormal(n0,n1,n2)), 0.0f);
        float triAreaPDF = 1.0f / Triangle::GetTriangleArea(p0,p1,p2);  //  probably could just precompute the triangle's area but that is one more float or two to store per triangle, need to test for memory cost vs performance benefits.
        float totalPDF = sampledLight.pmf * triAreaPDF * (distance * distance);
        
        sampleThroughput *= brdf * cosTheta_x * cosTheta_y / totalPDF;
        
        sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-12f;
        sampleRay.direction = newDir;
        
        samplePayload = TraceRay(sampleRay, activeScene);

        // Miss: hit sky
        if (samplePayload.hitDistance < 0.0f)
        {
            radiance += sampleThroughput * settings.skyColor;
            continue;
        }

        //  check if ray actually hits light source
        if(static_cast<uint32_t>(samplePayload.objectIndex) != sampledLight.emitterIndex)
            continue;   //  if not visible then return no radiance
        
        const Triangle& tri = activeScene->triangles[sampledLight.emitterIndex];
        const Material& material = activeScene->materials[tri.materialIndex];

        // Hit emissive
        glm::vec3 emission = material.GetEmission();
        float emmisiveRadiance = material.GetEmissionRadiance();
        if (emmisiveRadiance > 0.0f)
            radiance += sampleThroughput * emission;
    }

    radiance /= float(sampleCount);
    return glm::vec4(radiance, 1.0f);
}

//  NEXT EVENT ESTIMATION
__host__ __device__ glm::vec4 RendererGPU::PerPixel_NextEventEstimation(
    uint32_t x, uint32_t y,
    uint8_t maxBounces, uint8_t sampleCount,
    uint32_t frameIndex, const RenderingSettings& settings,
    const Scene_GPU* activeScene, const Camera_GPU* activeCamera,
    uint32_t imageWidth)
{
    uint32_t seed = (x + y * imageWidth) * frameIndex;
    glm::vec3 radiance{0.0f};

    // PRIMARY RAY
    Ray ray;
    ray.origin    = activeCamera->position;
    ray.direction = activeCamera->rayDirections[x + y * imageWidth];

    RayHitPayload payload = TraceRay(ray, activeScene);

    //  Skybox
    if (payload.hitDistance < 0.0f)
        return glm::vec4(settings.skyColor, 1.0f);
    
    const Triangle& hitTri = activeScene->triangles[payload.objectIndex];
    const Material& hitMat = activeScene->materials[hitTri.materialIndex];

    // Hit emissive surface
    if (glm::length(hitMat.GetEmission()) > 0.0f)
        return glm::vec4(hitMat.GetEmission(), 1.0f);
    
    // Multi-sample per pixel
    for (int s = 0; s < sampleCount; ++s)
    {
        seed += (s + 1) * 31;   // original seed update

        glm::vec3 pathThroughput{1.0f};
        Ray       pathRay      = ray;
        RayHitPayload hit      = payload;
        
        float pdfBRDF = 1.0f, pdfDirect = 1.0f;
        float weightBRDF = 1.0f, weightDirect = 1.0f;

        // Bounce loop
        for (int bounce = 0; bounce < maxBounces; ++bounce)
        {
            const Triangle& tri = activeScene->triangles[hit.objectIndex];
            const Material& mat = activeScene->materials[tri.materialIndex];
            
            // -------------------------------
            //   DIRECT LIGHT 
            // -------------------------------

            LightTree::ShadingPointQuery sp;
                sp.normal   = hit.worldNormal;
                sp.position = hit.worldPosition;

                LightTree::SampledLight sampled = PickLight_TLAS(activeScene->meshes, activeScene->lightTree_tlas, sp, seed);

                // light triangle data
                const Triangle& lTri = activeScene->triangles[sampled.emitterIndex];
                glm::vec3 p0 = activeScene->worldVertices[lTri.v0].position;
                glm::vec3 p1 = activeScene->worldVertices[lTri.v1].position;
                glm::vec3 p2 = activeScene->worldVertices[lTri.v2].position;
                glm::vec3 n0 = activeScene->worldVertices[lTri.v0].normal;
                glm::vec3 n1 = activeScene->worldVertices[lTri.v1].normal;
                glm::vec3 n2 = activeScene->worldVertices[lTri.v2].normal;

                glm::vec3 lightPoint = Triangle::GetRandomPointOnTriangle(p0,p1,p2,seed);
                glm::vec3 lightDir = lightPoint - hit.worldPosition;
                float dist = glm::length(lightDir);
                lightDir /= dist;

                // Shadow ray
                Ray shadowRay;
                shadowRay.origin = hit.worldPosition + hit.worldNormal * 1e-12f;
                shadowRay.direction = lightDir;

                RayHitPayload shadowPayload = TraceRay(shadowRay, activeScene);

                // Only add if unoccluded and hit correct emitter
                if (shadowPayload.hitDistance > 0.0f &&
                    static_cast<uint32_t>(shadowPayload.objectIndex) == sampled.emitterIndex)
                {
                    glm::vec3 lightNormal = Triangle::GetTriangleNormal(n0,n1,n2);

                    glm::vec3 brdf = MathUtils::CalculateBRDF(
                        hit.worldNormal,
                        -pathRay.direction,
                        lightDir,
                        mat.albedo,
                        mat.metallic,
                        mat.roughness
                    );

                    float cosTheta_x = glm::max(glm::dot(lightDir, hit.worldNormal), 0.0f);
                    float cosTheta_y = glm::max(glm::dot(-lightDir, lightNormal), 1e-12f);

                    // triangle area pdf (area measure)
                    float triArea = Triangle::GetTriangleArea(p0,p1,p2);
                    float triAreaPDF = 1.0f / triArea; // p_A

                    // convert area PDF -> solid-angle PDF:
                    // p_ω = p_A * r^2 / cosTheta_y
                    float lightSolidAnglePDF = triAreaPDF * (dist * dist) / cosTheta_y;

                    // get all the probabilily of directly choosing light source and probabily of choosing that direction according to BRDF
                    pdfDirect = sampled.pmf * lightSolidAnglePDF;
                    pdfBRDF = MathUtils::BRDFHemispherePDF(hit.worldNormal, -pathRay.direction, lightDir, mat.albedo, mat.metallic, mat.roughness);

                    //  Do MIS weighting
                    //  calc balance heuristic
                    weightBRDF = pdfBRDF / glm::max(pdfBRDF + pdfDirect, 1e-12f);
                    weightDirect = 1.0f - weightBRDF;
                    
                    const Material& lightMat = activeScene->materials[lTri.materialIndex];
                    
                    radiance += weightDirect *
                                pathThroughput *
                                brdf *
                                cosTheta_x *
                                lightMat.GetEmission() /
                                pdfDirect;
                }

            //  since not doing GI, dont bother continue
            if(maxBounces == 1)
            {
                radiance /= weightDirect;   //  undo MIS so it is exactly similiar like regular Light Source Sampling
                break;
            }
                
            
            // -------------------------------
            //   INDIRECT BOUNCE
            // -------------------------------
            glm::vec3 nextDir = MathUtils::BRDFSampleHemisphere(
                hit.worldNormal,
                -pathRay.direction,
                mat.albedo,
                mat.metallic,
                mat.roughness,
                seed,
                pdfBRDF 
            );

            pdfBRDF = glm::max(pdfBRDF, 1e-12f);
            
            glm::vec3 brdf = MathUtils::CalculateBRDF(
                hit.worldNormal,
                -pathRay.direction,
                nextDir,
                mat.albedo,
                mat.metallic,
                mat.roughness
            );

            float cosTheta = glm::dot(nextDir, hit.worldNormal);

            // throughput update
            pathThroughput *=  brdf * cosTheta / pdfBRDF;

            pathRay.origin    = hit.worldPosition + hit.worldNormal * 1e-12f;
            pathRay.direction = nextDir;

            // Trace next bounce
            hit = TraceRay(pathRay, activeScene);
            sp.normal   = hit.worldNormal;
            sp.position = hit.worldPosition;
            
            //  if brdf sampling hits light source
            //  Skybox
            if (hit.hitDistance < 0.0f)
            {
                radiance += pathThroughput * settings.skyColor;
                break;
            }

            // Emissive surface
            const Triangle& hitEmissiveTri = activeScene->triangles[hit.objectIndex];
            const Material& hitEmissiveMat = activeScene->materials[hitEmissiveTri.materialIndex];
            glm::vec3 emission = hitEmissiveMat.GetEmission();
            if (hitEmissiveMat.GetEmissionRadiance() > 0.0f)
            {
                glm::vec3 p0 = activeScene->worldVertices[hitEmissiveTri.v0].position;
                glm::vec3 p1 = activeScene->worldVertices[hitEmissiveTri.v1].position;
                glm::vec3 p2 = activeScene->worldVertices[hitEmissiveTri.v2].position;
                glm::vec3 n0 = activeScene->worldVertices[hitEmissiveTri.v0].normal;
                glm::vec3 n1 = activeScene->worldVertices[hitEmissiveTri.v1].normal;
                glm::vec3 n2 = activeScene->worldVertices[hitEmissiveTri.v2].normal;

                glm::vec3 lightPoint = Triangle::GetRandomPointOnTriangle(p0,p1,p2,seed);
                glm::vec3 lightDir = lightPoint - hit.worldPosition;
                float dist = glm::length(lightDir);
                lightDir /= dist;
                glm::vec3 lightNormal = Triangle::GetTriangleNormal(n0,n1,n2);
                float cosTheta_y = glm::max(glm::dot(-lightDir, lightNormal), 1e-12f);
                
                float triArea = Triangle::GetTriangleArea(p0,p1,p2);
                float triAreaPDF = 1.0f / triArea; // p_A

                // convert area PDF -> solid-angle PDF:
                float lightSolidAnglePDF = triAreaPDF * (dist * dist) / cosTheta_y;
                pdfDirect = ComputeDirectEmitterPMF(activeScene->meshes, activeScene->lightTree_tlas, sp, hit.objectIndex);
                pdfDirect *= lightSolidAnglePDF;
                //  Do MIS weighting
                //  calc balance heuristic
                weightBRDF = pdfBRDF / glm::max(pdfBRDF + pdfDirect, 1e-12f);
                radiance += weightBRDF * pathThroughput * emission;
                break;
            }
            
        } // bounce loop
    }     // sample loop

    return glm::vec4(radiance / float(sampleCount), 1.0f);
}

__host__ __device__ RayHitPayload RendererGPU::ClosestHit(
    const Ray& ray, float hitDistance, int objectIndex,
    float u, float v, const Scene_GPU* activeScene)
{
    RayHitPayload payload;
    payload.hitDistance = hitDistance;
    payload.objectIndex = objectIndex;

    const Triangle& tri = activeScene->triangles[objectIndex];
    payload.worldPosition = ray.origin + ray.direction * hitDistance;

    // Interpolate normals
    float w = 1.0f - u - v;
    glm::vec3 n0 = activeScene->worldVertices[tri.v0].normal;
    glm::vec3 n1 = activeScene->worldVertices[tri.v1].normal;
    glm::vec3 n2 = activeScene->worldVertices[tri.v2].normal;

    payload.worldNormal = glm::normalize(n0 * w + n1 * u + n2 * v);

    return payload;
}

__host__ __device__ RayHitPayload RendererGPU::Miss(const Ray& ray)
{
    RayHitPayload payload;
    payload.hitDistance = -1.0f;
    return payload;
}

__global__ void RenderKernel(glm::vec4* accumulationData, uint32_t* renderImageData, uint32_t width, uint32_t height, uint32_t frameIndex, RenderingSettings settings, const Scene_GPU* scene, const Camera_GPU* camera)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    size_t index = x + y * width;

    glm::vec4 pixelColor{0.0f};
    switch(settings.currentSamplingTechnique)
    {
    case BRUTE_FORCE:
        pixelColor = RendererGPU::PerPixel_BruteForce(x, y, static_cast<uint8_t>(settings.lightBounces), static_cast<uint8_t>(settings.sampleCount), frameIndex, settings, scene, camera, width);
        break;
    case UNIFORM_SAMPLING:
        pixelColor = RendererGPU::PerPixel_UniformSampling(x, y, static_cast<uint8_t>(settings.lightBounces), static_cast<uint8_t>(settings.sampleCount), frameIndex, settings, scene, camera, width);
        break;
    case COSINE_WEIGHTED_SAMPLING:
        pixelColor = RendererGPU::PerPixel_CosineWeightedSampling(x, y, static_cast<uint8_t>(settings.lightBounces), static_cast<uint8_t>(settings.sampleCount), frameIndex, settings, scene, camera, width);
        break;
    case GGX_SAMPLING:
        pixelColor = RendererGPU::PerPixel_GGXSampling(x, y, static_cast<uint8_t>(settings.lightBounces), static_cast<uint8_t>(settings.sampleCount), frameIndex, settings, scene, camera, width);
        break;
    case BRDF_SAMPLING:
        pixelColor = RendererGPU::PerPixel_BRDFSampling(x, y, static_cast<uint8_t>(settings.lightBounces), static_cast<uint8_t>(settings.sampleCount), frameIndex, settings, scene, camera, width);
        break;
    case LIGHT_SOURCE_SAMPLING:
        pixelColor = RendererGPU::PerPixel_LightSourceSampling(x, y, static_cast<uint8_t>(settings.lightBounces), static_cast<uint8_t>(settings.sampleCount), frameIndex, settings, scene, camera, width);
        break;
    case NEE:
        pixelColor = RendererGPU::PerPixel_NextEventEstimation(x, y, static_cast<uint8_t>(settings.lightBounces), static_cast<uint8_t>(settings.sampleCount), frameIndex, settings, scene, camera, width);
        break;
    //case RESTIR_DI:
    //case RESTIR_GI:
    default:
        pixelColor = RendererGPU::PerPixel_BruteForce(x, y, static_cast<uint8_t>(settings.lightBounces), static_cast<uint8_t>(settings.sampleCount), frameIndex, settings, scene, camera, width);
    }
    

    
    // Prevent NaNs or Infs from propagating
    if (!glm::all(glm::isfinite(pixelColor)))
        pixelColor = glm::vec4(0.0f);

    // Accumulate pixel color
    accumulationData[index] += pixelColor;

    // Average over frames
    glm::vec4 accumulatedColor = accumulationData[index] / (float)frameIndex;

    // Simple tone mapping for HDR
    accumulatedColor = accumulatedColor / (accumulatedColor + glm::vec4(1,1,1,0));

    // Clamp to valid range
    accumulatedColor = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0f));

    // Convert to packed RGBA
    renderImageData[index] = ColorUtils::ConvertToRGBA(accumulatedColor);
}
