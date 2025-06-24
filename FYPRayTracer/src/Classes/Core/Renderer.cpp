#include "Renderer.h"
#include "../../Utility/ColorUtils.h"
#include <execution>

RayHitPayload Renderer::TraceRay(const Ray& ray)   //  Project a ray per pixel to determine pixel output
{
    if(m_ActiveScene->spheres.empty())
        return Miss(ray);

    int closestSphere = -1;
    float hitDistance = FLT_MAX;

    //  find closest sphere and draw it 
    for(uint32_t i = 0; i <  m_ActiveScene->spheres.size(); i++)
    {
        const Sphere& sphere = m_ActiveScene->spheres[i];
        ////  Sphere Ray Hit Detection
        //  (bx^2 + by^2)t^2 + (2(axbx + ayby))t + (ax^2 + ay^2 - r^2) = 0
        //  similar to ax^2 + bx + c = 0
        //  where
        //  a = ray origin
        //  b = ray direction
        //  r = circle radius
        //  t = hit distance
    
        //  subtract with sphere's pos to take account of its position transform, formula above haven't yet take account of sphere position
        glm::vec3 origin = ray.origin - sphere.position;
    
        float a = glm::dot(ray.direction, ray.direction); //same as a = rayDirection.x * rayDirection.x + rayDirection.y * rayDirection.y + rayDirection.z * rayDirection.z;
        float b = 2.0f * glm::dot(origin, ray.direction);
        float c = glm::dot(origin,origin) - sphere.radius * sphere.radius;

        //  The discriminant of the quadratic formula
        //  b^2 - 4ac
        //  Used to determine if there is a ray hit to the sphere
        float discriminant = b * b - 4.0f * a * c;
        if(discriminant < 0.0f)
            continue;

        //  Quadratic formula
        //  -b +- sqrt(discriminant) / 2a
        //float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a);
        float closestHit = (-b - glm::sqrt(discriminant)) / (2.0f * a);   //  since a will always be positive, the minus part of the formula will always be smaller so thus closer to ray origin
        if(closestHit < hitDistance && closestHit > 0.0f)
        {
            hitDistance = closestHit;
            closestSphere = (int)i;
        }
    }

    if(closestSphere < 0)
        return Miss(ray);

    return ClosestHit(ray, hitDistance, closestSphere);
}

glm::vec4 Renderer::PerPixel(const uint32_t& x, const uint32_t& y, const uint8_t& bounces, const uint8_t& sampleCount)
{
    uint32_t seed = x + y * m_FinalRenderImage->GetWidth();
    seed *= m_FrameIndex;

    Ray ray;
    ray.origin = m_ActiveCamera->GetPosition();
    ray.direction = m_ActiveCamera->GetRayDirections()[x + y * m_FinalRenderImage->GetWidth()];

    glm::vec3 radiance{0.0f};
    glm::vec3 contribution{1.0f};

    
    const uint8_t maxBounces = bounces;

    for (int bounce = -1; bounce < maxBounces; bounce++)
    {
        seed += (uint32_t)(bounce+1);
        
        RayHitPayload payload = TraceRay(ray);

        //  if hit skybox
        if (payload.hitDistance < 0.0f)
        {
            radiance += contribution * m_Settings.skyColor; // Skybox color
            break;
        }

        const Sphere& sphere = m_ActiveScene->spheres[payload.objectIndex];
        const Material& material = m_ActiveScene->materials[sphere.materialIndex];

        //  if hit light source
        glm::vec3 emission = material.GetEmission();
        if (glm::length(emission) > 0.0f)
        {
            radiance += contribution * emission;
            break;
        }

        // Final bounce: project multiple rays and average the results
        if (bounce == maxBounces)
        {
            glm::vec3 finalBounceColor{0.0f};
            for (uint8_t i = 0; i < sampleCount; i++)
            {
                seed += (uint32_t)i;
                glm::vec3 newDir = MathUtils::CosineSampleHemisphere(payload.worldNormal, seed);
                float cosTheta = glm::dot(newDir, payload.worldNormal);
                if (cosTheta <= 0.0f) continue;

                float pdf = MathUtils::CosineHemispherePDF(cosTheta);

                glm::vec3 brdf = CalculateBRDF(
                    payload.worldNormal,
                    -ray.direction,  // V
                    newDir,          // L
                    material.albedo,
                    material.metallic,
                    material.roughness
                );

                Ray bounceRay;
                bounceRay.origin = payload.worldPosition + payload.worldNormal * 1e-3f;
                bounceRay.direction = glm::normalize(newDir);

                RayHitPayload bouncePayload = TraceRay(bounceRay);
                if (bouncePayload.hitDistance < 0.0f)
                {
                    finalBounceColor += brdf * m_Settings.skyColor * cosTheta / pdf;
                }
                else
                {
                    const Material& bounceMat = m_ActiveScene->materials[
                        m_ActiveScene->spheres[bouncePayload.objectIndex].materialIndex];
                    finalBounceColor += brdf * bounceMat.GetEmission() * cosTheta / pdf;
                }
            }
            finalBounceColor /= (float)sampleCount;
            radiance += contribution * finalBounceColor;
            break;
        }

        // Normal bounce path
        glm::vec3 newDir = MathUtils::CosineSampleHemisphere(payload.worldNormal, seed);
        float cosTheta = glm::dot(newDir, payload.worldNormal);
        if (cosTheta <= 0.0f) break;

        float pdf = MathUtils::CosineHemispherePDF(cosTheta);
        glm::vec3 brdf = CalculateBRDF(
            payload.worldNormal,
            -ray.direction,
            newDir,
            material.albedo,
            material.metallic,
            material.roughness
        );

        contribution *= brdf * cosTheta / pdf;
        ray.origin = payload.worldPosition + payload.worldNormal * 1e-3f;
        ray.direction = glm::normalize(newDir);
    }

    return glm::vec4(radiance, 1.0f);
}

RayHitPayload Renderer::ClosestHit(const Ray& ray, float hitDistance, int objectIndex)
{
    RayHitPayload payload;
    payload.hitDistance = hitDistance;
    payload.objectIndex = objectIndex;
    
    const Sphere& closestSphere = m_ActiveScene->spheres[objectIndex];
    
    glm::vec3 origin = ray.origin - closestSphere.position; //  disregard world position for easier calculations for now
    payload.worldPosition = origin + ray.direction * hitDistance;
    //To get our circle's normal vector on the hitPoint, normal = hitPoint - centerOfCircle. But since our circle right now is at origin, normal = hitPoint
    payload.worldNormal = glm::normalize(payload.worldPosition);

    payload.worldPosition += closestSphere.position;    //  move back to actual world position
    
    return payload;
}

RayHitPayload Renderer::Miss(const Ray& ray)
{
    RayHitPayload payload;
    payload.hitDistance = -1;
    return payload;
}

glm::vec3 Renderer::CalculateBRDF(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, const glm::vec3& albedo,
    float metallic, float roughness)
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
    glm::vec3 diffuse = (1.0f - metallic) * albedo / MathUtils::pi;

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


void Renderer::OnResize(uint32_t width, uint32_t height)
{
    if(m_FinalRenderImage)
    {
        //  no resize necessary
        if(m_FinalRenderImage->GetWidth() == width && m_FinalRenderImage->GetHeight() == height)
            return;

        m_FinalRenderImage->Resize(width, height); 
    }
    else
    {
        m_FinalRenderImage = std::make_shared<Walnut::Image>(width, height, Walnut::ImageFormat::RGBA);
    }

    delete[] m_RenderImageData;
    m_RenderImageData = new uint32_t[width * height];

    delete[] m_AccumulationData;
    m_AccumulationData = new glm::vec4[width * height];

    m_ImageHorizontalIter.resize(width);
    m_ImageVerticalIter.resize(height);
    for(uint32_t i = 0; i < width; i++)
        m_ImageHorizontalIter[i] = i;
    for(uint32_t i = 0; i < height; i++)
        m_ImageVerticalIter[i] = i;
}

void Renderer::Render(const Scene& scene, const Camera& camera)
{
    m_ActiveScene = &scene;
    m_ActiveCamera = &camera;

    constexpr size_t vec4Size = sizeof(glm::vec4);
    if(m_FrameIndex == 1)
        memset(m_AccumulationData, 0, m_FinalRenderImage->GetWidth() * m_FinalRenderImage->GetHeight() * vec4Size);
    
    //  draw every pixel onto screen


#define MT 1    //  set to 1 if we want CPU multithreading
#if MT
    std::for_each(std::execution::par, m_ImageVerticalIter.begin(), m_ImageVerticalIter.end(),
        [this](uint32_t y)
        {
            std::for_each(std::execution::par, m_ImageHorizontalIter.begin(), m_ImageHorizontalIter.end(),
                [this, y](uint32_t x)
                {
                  glm::vec4 pixelColor = PerPixel(x, y, (uint8_t)m_Settings.lightBounces, (uint8_t)m_Settings.sampleCount);
                  m_AccumulationData[x + y * m_FinalRenderImage->GetWidth()] += pixelColor;

                  glm::vec4 accumulatedColor = m_AccumulationData[x + y * m_FinalRenderImage->GetWidth()];
                  accumulatedColor /= (float)m_FrameIndex;

                  accumulatedColor = glm::clamp(accumulatedColor, glm::vec4{0.0f}, glm::vec4{1.0f});
                  m_RenderImageData[x + y * m_FinalRenderImage->GetWidth()] = ColorUtils::ConvertToRGBA(accumulatedColor);
                });
        });
#else
    for(uint32_t y = 0; y < m_FinalRenderImage->GetHeight(); y++)
    {
        for(uint32_t x = 0; x < m_FinalRenderImage->GetWidth(); x++)
        {
            glm::vec4 pixelColor = PerPixel(x, y);
            m_AccumulationData[x + y * m_FinalRenderImage->GetWidth()] += pixelColor;
    
            glm::vec4 accumulatedColor = m_AccumulationData[x + y * m_FinalRenderImage->GetWidth()];
            accumulatedColor /= (float)m_FrameIndex;
            
            accumulatedColor = glm::clamp(accumulatedColor, glm::vec4{0.0f}, glm::vec4{1.0f});
            m_RenderImageData[x + y * m_FinalRenderImage->GetWidth()] = ColorUtils::ConvertToRGBA(accumulatedColor);
        }
    }
#endif
    
    m_FinalRenderImage->SetData(m_RenderImageData);	//upload the image data onto GPU

    if(m_Settings.toAccumulate)
        m_FrameIndex++;
    else
        m_FrameIndex = 1;
}