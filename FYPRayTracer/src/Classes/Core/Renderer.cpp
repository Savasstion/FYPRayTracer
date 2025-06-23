#include "Renderer.h"
#include "Walnut/Random.h"
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

glm::vec4 Renderer::PerPixel(uint32_t x, uint32_t y)
{
    Ray ray;
    ray.origin = m_ActiveCamera->GetPosition();
    ray.direction = m_ActiveCamera->GetRayDirections()[x + y * m_FinalRenderImage->GetWidth()];

    glm::vec3 light{0.0f};
    glm::vec3 contribution{1.0f}; //  color of ray, will lose certain colors when materials absorb it
    
    int bounces = 5;
    for(int i = 0; i < bounces; i++)
    {
        RayHitPayload payload = TraceRay(ray);
        if(payload.hitDistance < 0.0f)
        {
            //glm::vec3 skyColor{0.6f,0.7f,0.9f};
            //light += skyColor * contribution;
            break;  //  stop tracing rays when there is nothing to bounce off of
        }
       
        const Sphere& sphere = m_ActiveScene->spheres[payload.objectIndex];
        const Material& material = m_ActiveScene->materials[sphere.materialIndex];
        
        // light will be absorbed by materials when it hit
        contribution *= material.albedo;
        //  add light emission from material
        light += material.GetEmission() * contribution;

        //  if hit light source already, no need to bounce again
        if(material.GetEmissionPower() > 0.0f)
            break;

        ray.origin = payload.worldPosition + payload.worldNormal * 0.000000001f;

        //  scuff implementation of projecting a ray within a hemisphere
        //  diffuse behavior, light will be scatter within a hemisphere
        ray.direction = glm::normalize(payload.worldNormal + Walnut::Random::InUnitSphere());
    }

    return {light, 1};  //  if hit, draw magenta pixel
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
                  glm::vec4 pixelColor = PerPixel(x, y);
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