#include "Renderer.h"
#include "RendererCUDA.cuh"
#include "../../Utility/ColorUtils.cuh"
#include <cuda.h>
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <iostream>

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
    Scene_GPU* d_sceneGPU = SceneToGPU(scene);   
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
    
    FreeSceneGPU(d_sceneGPU);
    FreeCameraGPU(d_cameraGPU);
    
}


__host__ __device__ RayHitPayload RendererGPU::TraceRay(const Ray& ray, const Scene_GPU* activeScene)
{
    if (activeScene->triangleCount == 0) return Miss(ray);

    float closestHitDistance = FLT_MAX;
    int closestTriangle = -1;
    float closestU = 0.0f;
    float closestV = 0.0f;

     // // Loop over GPU triangles (old way to do ray triangle intersect)
     //  for (size_t objectIndex = 0; objectIndex < activeScene->triangleCount; objectIndex++)
     //  {
     //      const Triangle& triangle = activeScene->triangles[objectIndex];
     //
     //      const glm::vec3& v0 = activeScene->worldVertices[triangle.v0].position;
     //      const glm::vec3& v1 = activeScene->worldVertices[triangle.v1].position;
     //      const glm::vec3& v2 = activeScene->worldVertices[triangle.v2].position;
     //
     //      glm::vec3 edge1 = v1 - v0;
     //      glm::vec3 edge2 = v2 - v0;
     //      glm::vec3 h = glm::cross(ray.direction, edge2);
     //      float a = glm::dot(edge1, h);
     //
     //      float absoluteOfA = a < 0.0f ? -a : a;
     //      if (absoluteOfA < 1e-8f) continue;
     //
     //      float f = 1.0f / a;
     //      glm::vec3 s = ray.origin - v0;
     //      float u = f * glm::dot(s, h);
     //      if (u < 0.0f || u > 1.0f) continue;
     //
     //      glm::vec3 q = glm::cross(s, edge1);
     //      float v = f * glm::dot(ray.direction, q);
     //      if (v < 0.0f || u + v > 1.0f) continue;
     //
     //      float t = f * glm::dot(edge2, q);
     //      if (t > 0.0001f && t < closestHitDistance)
     //      {
     //          closestHitDistance = t;
     //          closestTriangle = static_cast<int>(objectIndex);
     //          closestU = u;
     //          closestV = v;
     //      }
     //  }

    // //BVH Traversal
    // BVH* tlas = activeScene->tlas;
    //
    // const int TLAS_STACK_SIZE = 1024;
    // int tlasStack[TLAS_STACK_SIZE];
    // int tlasPointer = 0;
    // tlasStack[tlasPointer++] = tlas->rootIndex;
    //
    // while (tlasPointer > 0)
    // {
    //     int nodeIndex = tlasStack[--tlasPointer];
    //     
    //     if (!IntersectRayAABB(ray, tlas->nodes[nodeIndex].box))
    //         continue;
    //
    //     if (tlas->nodes[nodeIndex].isLeaf)
    //     {
    //         const Triangle& triangle = activeScene->triangles[tlas->nodes[nodeIndex].objectIndex];
    //         
    //         const glm::vec3& v0 = activeScene->worldVertices[triangle.v0].position;
    //         const glm::vec3& v1 = activeScene->worldVertices[triangle.v1].position;
    //         const glm::vec3& v2 = activeScene->worldVertices[triangle.v2].position;
    //         
    //         glm::vec3 edge1 = v1 - v0;
    //         glm::vec3 edge2 = v2 - v0;
    //         glm::vec3 h = glm::cross(ray.direction, edge2);
    //         float a = glm::dot(edge1, h);
    //         
    //         float absoluteOfA = a < 0.0f ? -a : a;
    //         if (absoluteOfA < 1e-8f) continue;
    //         
    //         float f = 1.0f / a;
    //         glm::vec3 s = ray.origin - v0;
    //         float u = f * glm::dot(s, h);
    //         if (u < 0.0f || u > 1.0f) continue;
    //         
    //         glm::vec3 q = glm::cross(s, edge1);
    //         float v = f * glm::dot(ray.direction, q);
    //         if (v < 0.0f || u + v > 1.0f) continue;
    //         
    //         float t = f * glm::dot(edge2, q);
    //         if (t > 0.0001f && t < closestHitDistance)
    //         {
    //             closestHitDistance = t;
    //             closestTriangle = static_cast<int>(tlas->nodes[nodeIndex].objectIndex);
    //             closestU = u;
    //             closestV = v;
    //         }
    //     }
    //     else
    //     {
    //         tlasStack[tlasPointer++] = tlas->nodes[nodeIndex].child1;
    //         tlasStack[tlasPointer++] = tlas->nodes[nodeIndex].child2;
    //     }
    // }

    //TLAS - BLAS Traversal
     BVH* tlas = activeScene->tlas;
    
     const int TLAS_STACK_SIZE = 1024;
     int tlasStack[TLAS_STACK_SIZE];
     int tlasStackTop = 0;
     tlasStack[tlasStackTop++] = static_cast<int>(tlas->rootIndex);
    
     while (tlasStackTop > 0)
     {
         int nodeIndex = tlasStack[--tlasStackTop];
    
         const BVH::Node& node = tlas->nodes[nodeIndex];
         if (!IntersectRayAABB(ray, node.box)) continue;

         // test
         if (nodeIndex == 1)
         {
             nodeIndex = 1;
         }
    
         if (node.isLeaf)
         {
             size_t meshIndex = node.objectIndex;
             
             BVH* blas = activeScene->meshes[meshIndex].blas;

             //debug check
             //auto n0 = blas->nodes[0];
             //auto n1 = blas->nodes[1];
             //auto n2 = blas->nodes[2];
             //auto n3 = blas->nodes[3];
             //auto n4 = blas->nodes[4];
             //auto n5 = blas->nodes[5];
             //auto n6 = blas->nodes[6];
    
             const int BLAS_STACK_SIZE = 2048;
             int blasStack[BLAS_STACK_SIZE];
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
    
    
                     glm::vec3 edge1 = v1 - v0;
                     glm::vec3 edge2 = v2 - v0;
                     glm::vec3 h = glm::cross(ray.direction, edge2);
                     float a = glm::dot(edge1, h);
                     float absA = a < 0.0f ? -a : a;
                     if (absA < 1e-8f) continue;
    
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
__host__ __device__ glm::vec4 RendererGPU::PerPixel(
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

        glm::vec3 brdf = CalculateBRDF(
            primaryPayload.worldNormal,
            -primaryRay.direction,
            newDir,
            hitMaterial.albedo,
            hitMaterial.metallic,
            hitMaterial.roughness
        );

        float cosTheta = glm::max(glm::dot(newDir, primaryPayload.worldNormal), 0.0f);
        sampleThroughput *= (brdf * cosTheta / glm::max(pdf, 1e-4f));

        sampleRay.origin = primaryPayload.worldPosition + primaryPayload.worldNormal * 1e-3f;
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

    radiance /= float(sampleCount);
    return glm::vec4(radiance, 1.0f);
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

__global__ void RenderKernel(
    glm::vec4* accumulationData,
    uint32_t* renderImageData,
    uint32_t width,
    uint32_t height,
    uint32_t frameIndex,
    RenderingSettings settings,
    const Scene_GPU* scene,
    const Camera_GPU* camera
)
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

    size_t index = x + y * width;

    // Accumulate pixel color
    accumulationData[index] += pixelColor;

    // Average over frames
    glm::vec4 accumulatedColor = accumulationData[index] / (float)frameIndex;

    // Clamp to valid range
    accumulatedColor = glm::clamp(accumulatedColor, glm::vec4(0.0f), glm::vec4(1.0f));

    // Convert to packed RGBA
    renderImageData[index] = ColorUtils::ConvertToRGBA(accumulatedColor);
}
