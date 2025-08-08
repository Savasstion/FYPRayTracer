#include "Renderer.h"
#include "RendererCUDA.cuh"
#include "../../Utility/ColorUtils.cuh"
#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>

void Renderer::Render(const Scene& scene, const Camera& camera)
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
    cudaMalloc(&d_accumulationData, pixelCount * vec4Size);
    cudaMalloc(&d_renderImageData, pixelCount * uint32Size);

    // Initialize or copy accumulation buffer
    if (m_FrameIndex == 1)
    {
        cudaMemset(d_accumulationData, 0, pixelCount * vec4Size);
    }
    else
    {
        // Ensure m_AccumulationData is valid host pointer and initialized
        cudaMemcpy(d_accumulationData, m_AccumulationData, pixelCount * vec4Size, cudaMemcpyHostToDevice);
    }
    
    // For demonstration, pass host pointers as nullptr (avoid crash)
    const Scene* d_scene = nullptr;
    const Camera* d_camera = nullptr;

    // Copy settings to device (small struct, okay)
    Settings h_settings = m_Settings;
    Settings* d_settings;
    cudaMalloc(&d_settings, sizeof(Settings));
    cudaMemcpy(d_settings, &h_settings, sizeof(Settings), cudaMemcpyHostToDevice);

    // Correct thread/block sizes
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(
        (width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    RenderKernel<<<numBlocks, threadsPerBlock>>>(
        d_accumulationData,
        d_renderImageData,
        width,
        height,
        m_FrameIndex,
        *d_settings,
        d_scene,
        d_camera);
    
    cudaDeviceSynchronize();
    
    // Copy results back (make sure host buffers are valid and allocated)
    cudaMemcpy(m_AccumulationData, d_accumulationData, pixelCount * vec4Size, cudaMemcpyDeviceToHost);
    cudaMemcpy(m_RenderImageData, d_renderImageData, pixelCount * uint32Size, cudaMemcpyDeviceToHost);

    m_FinalRenderImage->SetData(m_RenderImageData);

    if (m_Settings.toAccumulate)
        m_FrameIndex++;
    else
        m_FrameIndex = 1;

    cudaFree(d_accumulationData);
    cudaFree(d_renderImageData);
    cudaFree(d_settings);
}


__host__ __device__ RayHitPayload RendererGPU::TraceRay(const Ray& ray,const Scene* activeScene)   //  Project a ray per pixel to determine pixel output
{
    if (activeScene->triangles.empty())
        return Miss(ray);

    int closestTriangle = -1;
    float hitDistance = FLT_MAX;
    float closestU = 0.0f;
    float closestV = 0.0f;

    // Find closest triangle and draw it
    // NOTE: BVH traversal must be __device__ too
    //       or replaced with GPU-friendly logic.
    // This call won't work in CUDA unless BVH is also GPU compatible.
    // Placeholder: you’ll need a GPU BVH traversal here.
    // activeScene->bvh.TraverseRayRecursiveGPU(...)

    for (size_t objectIndex = 0; objectIndex < activeScene->triangles.size(); objectIndex++)
    {
        const Triangle& triangle = activeScene->triangles[objectIndex];

        const glm::vec3& v0 = activeScene->worldVertices[triangle.v0].position;
        const glm::vec3& v1 = activeScene->worldVertices[triangle.v1].position;
        const glm::vec3& v2 = activeScene->worldVertices[triangle.v2].position;

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;
        glm::vec3 h = glm::cross(ray.direction, edge2);
        float a = glm::dot(edge1, h);

        float absoluteOfA = a < 0.0f ? -a : a;
        if (absoluteOfA < 1e-8f) continue;

        float f = 1.0f / a;
        glm::vec3 s = ray.origin - v0;
        float u = f * glm::dot(s, h);
        if (u < 0.0f || u > 1.0f) continue;

        glm::vec3 q = glm::cross(s, edge1);
        float v = f * glm::dot(ray.direction, q);
        if (v < 0.0f || u + v > 1.0f) continue;

        float t = f * glm::dot(edge2, q);
        if (t > 0.0001f && t < hitDistance)
        {
            hitDistance = t;
            closestTriangle = static_cast<int>(objectIndex);
            closestU = u;
            closestV = v;
        }
    }

    if (closestTriangle < 0)
        return Miss(ray);

    return ClosestHit(ray, hitDistance, closestTriangle, closestU, closestV, activeScene);
}

//  //  PURE BRUTE-FORCE
// glm::vec4 Renderer::PerPixel(const uint32_t& x, const uint32_t& y, const uint8_t& maxBounces, const uint8_t& sampleCount)
// {
//     //  PURE BRUTE-FORCE, EVERY FRAME ONLY CHOOSE ONE RANDOM PATH TO FOLLOW
//     //  LIKE A OFFLINE-RENDERER, KEEPS TRACK OF SUM AND THEN AVERAGE THE PIXEL'S COLOR OVER TIME TO EVENTUALLY FORM A PHYSICALLY-ACCURATE IMAGE (keyword : "eventually")
//     
//     uint32_t seed = x + y * m_FinalRenderImage->GetWidth();
//     seed *= m_FrameIndex;
//
//     Ray ray;
//     ray.origin = m_ActiveCamera->GetPosition();
//     ray.direction = m_ActiveCamera->GetRayDirections()[x + y * m_FinalRenderImage->GetWidth()];
//
//     glm::vec3 radiance{0.0f};
//     glm::vec3 contribution{1.0f};
//
//     
//     const uint8_t rayProjectCount = maxBounces + 1; //  add one to ensure first camera ray is still projected
//
//     for (int currentBounce = 0; currentBounce < rayProjectCount; currentBounce++)
//     {
//         seed += (uint32_t)(currentBounce+1);
//         
//         RayHitPayload payload = TraceRay(ray);
//
//         //  if hit skybox
//         if (payload.hitDistance < 0.0f)
//         {
//             radiance += contribution * m_Settings.skyColor; // Skybox color
//             break;
//         }
//
//         const Sphere& sphere = m_ActiveScene->spheres[payload.objectIndex];
//         const Material& material = m_ActiveScene->materials[sphere.materialIndex];
//
//         //  if hit light source
//         glm::vec3 emission = material.GetEmission();
//         if (glm::length(emission) > 0.0f)
//         {
//             radiance += contribution * emission;
//             break;
//         }
//
//         glm::vec3 newDir = MathUtils::UniformSampleHemisphere(payload.worldNormal, seed);
//         float cosTheta = glm::dot(newDir, payload.worldNormal); //  Geometry Term
//         if (cosTheta <= 0.0f) break;
//
//         float pdf = MathUtils::UniformHemispherePDF();
//         glm::vec3 brdf = CalculateBRDF(
//             payload.worldNormal,
//             -ray.direction,
//             newDir,
//             material.albedo,
//             material.metallic,
//             material.roughness
//         );
//
//         contribution *= brdf * cosTheta / pdf;
//         ray.origin = payload.worldPosition + payload.worldNormal * 1e-3f;
//         ray.direction = glm::normalize(newDir);
//     }
//
//     return glm::vec4(radiance, 1.0f);
// }

//  //  UNIFORM SAMPLING
// glm::vec4 Renderer::PerPixel(const uint32_t& x, const uint32_t& y, const uint8_t& maxBounces, const uint8_t& sampleCount)
// {
//     //  all directions within a hemisphere are all equally likely to be sampled
//     
//     uint32_t seed = x + y * m_FinalRenderImage->GetWidth();
//     seed *= m_FrameIndex;
//
//     glm::vec3 radiance{0.0f};   // Final color accumulated from all samples
//
//     // == PRIMARY RAY ==
//     Ray primaryRay;
//     primaryRay.origin = m_ActiveCamera->GetPosition();
//     primaryRay.direction = m_ActiveCamera->GetRayDirections()[x + y * m_FinalRenderImage->GetWidth()];
//
//     RayHitPayload primaryPayload = TraceRay(primaryRay);
//
//     // Hit sky immediately
//     if (primaryPayload.hitDistance < 0.0f)
//         return glm::vec4(m_Settings.skyColor, 1.0f);
//
//     const Sphere& hitSphere = m_ActiveScene->spheres[primaryPayload.objectIndex];
//     const Material& hitMaterial = m_ActiveScene->materials[hitSphere.materialIndex];
//
//     // Hit emissive object immediately
//     if (glm::length(hitMaterial.GetEmission()) > 0.0f)
//         return glm::vec4(hitMaterial.GetEmission(), 1.0f);
//
//     // == SAMPLE MULTIPLE LIGHT PATHS FROM FIRST HIT ==
//     for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
//     {
//         seed += (sampleIndex + 1) * 27;
//         glm::vec3 sampleThroughput{1.0f};
//         Ray sampleRay;
//         RayHitPayload samplePayload = primaryPayload;
//
//         // Sample initial direction from first hit
//         glm::vec3 newDir = MathUtils::UniformSampleHemisphere(primaryPayload.worldNormal, seed);
//
//         glm::vec3 brdf = CalculateBRDF(
//             primaryPayload.worldNormal,
//             -primaryRay.direction,
//             newDir,
//             hitMaterial.albedo,
//             hitMaterial.metallic,
//             hitMaterial.roughness
//         );
//
//         float cosTheta = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);  //  Geometry Term
//         float pdf = MathUtils::UniformHemispherePDF();
//         sampleThroughput *= (brdf * cosTheta / glm::max(pdf, 1e-4f)); // Rendering equation core
//
//         sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-3f;
//         sampleRay.direction = newDir;
//
//         // Trace the path for maxBounces
//         for (int bounce = 0; bounce < maxBounces; bounce++)
//         {
//             seed += sampleIndex + 31 * bounce;
//
//             samplePayload = TraceRay(sampleRay);
//
//             // Hit sky
//             if (samplePayload.hitDistance < 0.0f)
//             {
//                 radiance += sampleThroughput * m_Settings.skyColor;
//                 break;
//             }
//
//             const Sphere& sphere = m_ActiveScene->spheres[samplePayload.objectIndex];
//             const Material& material = m_ActiveScene->materials[sphere.materialIndex];
//
//             // Hit emissive light
//             glm::vec3 emission = material.GetEmission();
//             if (glm::length(emission) > 0.0f)
//             {
//                 radiance += sampleThroughput * emission;
//                 break;
//             }
//
//             // Sample next direction
//             glm::vec3 bounceDir = MathUtils::UniformSampleHemisphere(primaryPayload.worldNormal, seed);
//             
//             glm::vec3 bounceBrdf = CalculateBRDF(
//                 samplePayload.worldNormal,
//                 -sampleRay.direction,
//                 bounceDir,
//                 material.albedo,
//                 material.metallic,
//                 material.roughness
//             );
//
//             float bounceCosTheta = glm::max(glm::dot(bounceDir, samplePayload.worldNormal), 0.0f);
//             float bouncePdf = MathUtils::UniformHemispherePDF();
//             sampleThroughput *= bounceBrdf * bounceCosTheta / glm::max(bouncePdf, 1e-4f);
//
//             sampleRay.origin = samplePayload.worldPosition + samplePayload.worldNormal * 1e-3f;
//             sampleRay.direction = bounceDir;
//         }
//     }
//
//     radiance /= float(sampleCount); // Average across all sampled paths
//     return glm::vec4(radiance, 1.0f);
// }

//  //  COSINE-WEIGHTED SAMPLING
// glm::vec4 Renderer::PerPixel(const uint32_t& x, const uint32_t& y, const uint8_t& maxBounces, const uint8_t& sampleCount)
// {
//     //  a kind of BRDF sampling method that makes it so that shallower angles are less likely to be sampled as they likely contribute less light
//     
//     uint32_t seed = x + y * m_FinalRenderImage->GetWidth();
//     seed *= m_FrameIndex;
//
//     glm::vec3 radiance{0.0f};   // Final color accumulated from all samples
//
//     // == PRIMARY RAY ==
//     Ray primaryRay;
//     primaryRay.origin = m_ActiveCamera->GetPosition();
//     primaryRay.direction = m_ActiveCamera->GetRayDirections()[x + y * m_FinalRenderImage->GetWidth()];
//
//     RayHitPayload primaryPayload = TraceRay(primaryRay);
//
//     // Hit sky immediately
//     if (primaryPayload.hitDistance < 0.0f)
//         return glm::vec4(m_Settings.skyColor, 1.0f);
//
//     const Sphere& hitSphere = m_ActiveScene->spheres[primaryPayload.objectIndex];
//     const Material& hitMaterial = m_ActiveScene->materials[hitSphere.materialIndex];
//
//     // Hit emissive object immediately
//     if (glm::length(hitMaterial.GetEmission()) > 0.0f)
//         return glm::vec4(hitMaterial.GetEmission(), 1.0f);
//
//     // == SAMPLE MULTIPLE LIGHT PATHS FROM FIRST HIT ==
//     for (int sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++)
//     {
//         seed += (sampleIndex + 1) * 27;
//         glm::vec3 sampleThroughput{1.0f};
//         Ray sampleRay;
//         RayHitPayload samplePayload = primaryPayload;
//
//         // Sample initial direction from first hit
//         glm::vec3 newDir = MathUtils::CosineSampleHemisphere(primaryPayload.worldNormal, seed);
//
//         glm::vec3 brdf = CalculateBRDF(
//             primaryPayload.worldNormal,
//             -primaryRay.direction,
//             newDir,
//             hitMaterial.albedo,
//             hitMaterial.metallic,
//             hitMaterial.roughness
//         );
//
//         float cosTheta = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);  //  Geometry Term
//         float pdf = MathUtils::CosineHemispherePDF(cosTheta);
//         sampleThroughput *= (brdf * cosTheta / glm::max(pdf, 1e-4f)); // Rendering equation core
//
//         sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-3f;
//         sampleRay.direction = newDir;
//
//         // Trace the path for maxBounces
//         for (int bounce = 0; bounce < maxBounces; bounce++)
//         {
//             seed += sampleIndex + 31 * bounce;
//
//             samplePayload = TraceRay(sampleRay);
//
//             // Hit sky
//             if (samplePayload.hitDistance < 0.0f)
//             {
//                 radiance += sampleThroughput * m_Settings.skyColor;
//                 break;
//             }
//
//             const Sphere& sphere = m_ActiveScene->spheres[samplePayload.objectIndex];
//             const Material& material = m_ActiveScene->materials[sphere.materialIndex];
//
//             // Hit emissive light
//             glm::vec3 emission = material.GetEmission();
//             if (glm::length(emission) > 0.0f)
//             {
//                 radiance += sampleThroughput * emission;
//                 break;
//             }
//
//             // Sample next direction
//             glm::vec3 bounceDir = MathUtils::CosineSampleHemisphere(primaryPayload.worldNormal, seed);
//             
//             glm::vec3 bounceBrdf = CalculateBRDF(
//                 samplePayload.worldNormal,
//                 -sampleRay.direction,
//                 bounceDir,
//                 material.albedo,
//                 material.metallic,
//                 material.roughness
//             );
//
//             float bounceCosTheta = glm::max(glm::dot(bounceDir, samplePayload.worldNormal), 0.0f);
//             float bouncePdf = MathUtils::CosineHemispherePDF(bounceCosTheta);
//             sampleThroughput *= bounceBrdf * bounceCosTheta / glm::max(bouncePdf, 1e-4f);
//
//             sampleRay.origin = samplePayload.worldPosition + samplePayload.worldNormal * 1e-3f;
//             sampleRay.direction = bounceDir;
//         }
//     }
//
//     radiance /= float(sampleCount); // Average across all sampled paths
//     return glm::vec4(radiance, 1.0f);
// }

//  BRDF SAMPLING
__host__ __device__ glm::vec4 RendererGPU::PerPixel(uint32_t x, uint32_t y, uint8_t maxBounces, uint8_t sampleCount, uint32_t frameIndex, const Settings& settings, const Scene* activeScene, const Camera* activeCamera, uint32_t imageWidth)
{
    uint32_t seed = x + y * imageWidth;
    seed *= frameIndex;

    glm::vec3 radiance{0.0f};   // Final color accumulated from all samples

    // == PRIMARY RAY ==
    Ray primaryRay;
    primaryRay.origin = activeCamera->GetPosition();
    primaryRay.direction = activeCamera->GetRayDirections()[x + y * imageWidth];

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
        glm::vec3 newDir = MathUtils::BRDFSampleHemisphere(
            primaryPayload.worldNormal,
            -primaryRay.direction,
            hitMaterial.albedo,
            hitMaterial.metallic,
            hitMaterial.roughness,
            seed,
            pdf
        );

        glm::vec3 brdf = CalculateBRDF(
            primaryPayload.worldNormal,
            -primaryRay.direction,
            newDir,
            hitMaterial.albedo,
            hitMaterial.metallic,
            hitMaterial.roughness
        );

        float cosTheta = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);  // Geometry Term
        sampleThroughput *= (brdf * cosTheta / glm::max(pdf, 1e-4f)); // Rendering equation core

        sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-3f;
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
            glm::vec3 bounceDir = MathUtils::BRDFSampleHemisphere(
                samplePayload.worldNormal,
                -sampleRay.direction,
                material.albedo,
                material.metallic,
                material.roughness,
                seed,
                bouncePdf
            );

            glm::vec3 bounceBrdf = CalculateBRDF(
                samplePayload.worldNormal,
                -sampleRay.direction,
                bounceDir,
                material.albedo,
                material.metallic,
                material.roughness
            );

            float bounceCosTheta = glm::max(glm::dot(bounceDir, samplePayload.worldNormal), 0.0f);
            sampleThroughput *= bounceBrdf * bounceCosTheta / glm::max(bouncePdf, 1e-4f);

            sampleRay.origin = samplePayload.worldPosition + samplePayload.worldNormal * 1e-3f;
            sampleRay.direction = bounceDir;
        }
    }

    radiance /= float(sampleCount); // Average across all sampled paths
    return glm::vec4(radiance, 1.0f);
}

__host__ __device__ RayHitPayload RendererGPU::ClosestHit(const Ray& ray, float hitDistance, int objectIndex, float u, float v, const Scene* activeScene)
{
    RayHitPayload payload;
    payload.hitDistance = hitDistance;
    payload.objectIndex = objectIndex;

    // For Triangles
    const Triangle& tri = activeScene->triangles[objectIndex];

    payload.worldPosition = ray.origin + ray.direction * hitDistance;

    // Interpolate normal using barycentric coordinates
    float w = 1.0f - u - v;
    glm::vec3 n0 = activeScene->worldVertices[tri.v0].normal;
    glm::vec3 n1 = activeScene->worldVertices[tri.v1].normal;
    glm::vec3 n2 = activeScene->worldVertices[tri.v2].normal;

    glm::vec3 interpolatedNormal = glm::normalize(
        n0 * w +
        n1 * u +
        n2 * v
    );

    payload.worldNormal = interpolatedNormal;

    return payload;
}

__host__ __device__ RayHitPayload RendererGPU::Miss(const Ray& ray)
{
    RayHitPayload payload;
    payload.hitDistance = -1.0f;
    return payload;
}

__host__ __device__ glm::vec3 RendererGPU::CalculateBRDF(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, const glm::vec3& albedo, float metallic, float roughness)
{
    glm::vec3 H = glm::normalize(V + L);
    float NdotL = glm::max(glm::dot(N, L), 0.0f);
    float NdotV = glm::max(glm::dot(N, V), 0.0f);
    float NdotH = glm::max(glm::dot(N, H), 0.0f);
    float VdotH = glm::max(glm::dot(V, H), 0.0f);

    // Fresnel (Schlick's approximation)
    glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);
    glm::vec3 F = F0 + (1.0f - F0) * glm::pow(1.0f - VdotH, 5.0f);

    // Geometry Shadowing (Smith)
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    float G_V = NdotV / (NdotV * (1.0f - k) + k);
    float G_L = NdotL / (NdotL * (1.0f - k) + k);
    float G = G_V * G_L;
    
    // Lambertian diffuse (non-metallic only)
    glm::vec3 kD = (1.0f - F) * (1.0f - metallic);
    glm::vec3 diffuse = kD * albedo / MathUtils::pi;

    // Normal Distribution (GGX / Trowbridge-Reitz)
    float a = roughness * roughness;
    float a2 = a * a;
    float denominator = (NdotH * NdotH) * (a2 - 1.0f) + 1.0f;
    float D = a2 / (MathUtils::pi * denominator * denominator);

    
    //  Cook-Torrance specular
    glm::vec3 specular = (D * F * G) / glm::max(4.0f * NdotV * NdotL, 0.001f);

    //diffuse + specular should be max 1, if its above 1 then more energy is created than it should conserve
    return diffuse + specular;
}

__global__ void RenderKernel(glm::vec4* accumulationData, uint32_t* renderImageData, uint32_t width, uint32_t height, uint32_t frameIndex, Settings settings, const Scene* scene, const Camera* camera)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;
    
    glm::vec4 pixelColor = RendererGPU::PerPixel(
        x, 
        y, 
        (uint32_t)settings.lightBounces, 
        (uint32_t)settings.sampleCount, 
        frameIndex, 
        settings, 
        scene, 
        camera, 
        width
    );

    int index = x + y * width;

    // Accumulate pixel color
    accumulationData[index] += pixelColor;

    // Average over frames
    glm::vec4 accumulatedColor = accumulationData[index] / (float)frameIndex;

    // Clamp to valid range
    accumulatedColor = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0f));

    // Convert to packed RGBA
    renderImageData[index] = ColorUtils::ConvertToRGBA(accumulatedColor);
}

// void Renderer::Render(const Scene& scene, const Camera& camera)
// {
//     m_ActiveScene = &scene;
//     m_ActiveCamera = &camera;
//
//     constexpr size_t vec4Size = sizeof(glm::vec4);
//     if(m_FrameIndex == 1)
//         memset(m_AccumulationData, 0, m_FinalRenderImage->GetWidth() * m_FinalRenderImage->GetHeight() * vec4Size);
//     
//     //  draw every pixel onto screen
//
//
// #define MT 1    //  set to 1 if we want CPU multithreading
// #if MT
//     std::for_each(std::execution::par, m_ImageVerticalIter.begin(), m_ImageVerticalIter.end(),
//         [this](uint32_t y)
//         {
//             std::for_each(std::execution::par, m_ImageHorizontalIter.begin(), m_ImageHorizontalIter.end(),
//                 [this, y](uint32_t x)
//                 {
//                   glm::vec4 pixelColor = PerPixel(x, y, (uint8_t)m_Settings.lightBounces, (uint8_t)m_Settings.sampleCount, m_FrameIndex, m_Settings, m_ActiveScene, m_ActiveCamera, m_FinalRenderImage->GetWidth());
//                   m_AccumulationData[x + y * m_FinalRenderImage->GetWidth()] += pixelColor;
//
//                   glm::vec4 accumulatedColor = m_AccumulationData[x + y * m_FinalRenderImage->GetWidth()];
//                   accumulatedColor /= (float)m_FrameIndex;
//
//                   accumulatedColor = glm::clamp(accumulatedColor, glm::vec4{0.0f}, glm::vec4{1.0f});
//                   m_RenderImageData[x + y * m_FinalRenderImage->GetWidth()] = ColorUtils::ConvertToRGBA(accumulatedColor);
//                 });
//         });
// #else
//     for(uint32_t y = 0; y < m_FinalRenderImage->GetHeight(); y++)
//     {
//         for(uint32_t x = 0; x < m_FinalRenderImage->GetWidth(); x++)
//         {
//             glm::vec4 pixelColor = PerPixel(x, y);
//             m_AccumulationData[x + y * m_FinalRenderImage->GetWidth()] += pixelColor;
//     
//             glm::vec4 accumulatedColor = m_AccumulationData[x + y * m_FinalRenderImage->GetWidth()];
//             accumulatedColor /= (float)m_FrameIndex;
//             
//             accumulatedColor = glm::clamp(accumulatedColor, glm::vec4{0.0f}, glm::vec4{1.0f});
//             m_RenderImageData[x + y * m_FinalRenderImage->GetWidth()] = ColorUtils::ConvertToRGBA(accumulatedColor);
//         }
//     }
// #endif
//     
//     m_FinalRenderImage->SetData(m_RenderImageData);	//upload the image data onto GPU
//
//     if(m_Settings.toAccumulate)
//         m_FrameIndex++;
//     else
//         m_FrameIndex = 1;
// }