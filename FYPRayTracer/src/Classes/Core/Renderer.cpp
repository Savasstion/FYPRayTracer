#include "Renderer.h"
#include "Walnut/Random.h"
#include "../../Utility/ColorUtils.h"

glm::vec4 Renderer::TraceRay(const Scene& scene, const Ray& ray)   //  Project a ray per pixel to determine pixel output
{
    if(scene.spheres.empty())
        return {0,0, 0, 1}; //  if nothing in scene then don't bother

    const Sphere* closestSphere = nullptr;
    float hitDistance = FLT_MAX;

    //  find closest sphere and draw it 
    for(const Sphere& sphere : scene.spheres)
    {
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
        if(closestHit < hitDistance)    //  TODO: need to take account of near plane
        {
            hitDistance = closestHit;
            closestSphere = &sphere;
        }
    }

    if(closestSphere == nullptr)
        return {0,0, 0, 1};
    
    if(hitDistance < 0.0f)   //  if closest hit is negative, don't draw
        return {0,0, 0, 1};

    glm::vec3 origin = ray.origin - closestSphere->position;

    glm::vec3 hitPoint = origin + ray.direction * hitDistance;

    //To get our circle's normal vector onm the hitPoint, normal = hitPoint - centerOfCircle. But since our circle right now is at origin, normal = hitPoint
    glm::vec3 normal = glm::normalize(hitPoint);

    //  light direction
    glm::vec3 lightDir = glm::normalize(glm::vec3{-1,-1,-1});
    
    float d = glm::max(glm::dot(normal, -lightDir), 0.0f);  //  ==  cos(angle)
    
    glm::vec3 sphereColor = closestSphere->albedo;
    sphereColor *= d;
    return {sphereColor, 1};  //  if hit, draw magenta pixel
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
}

void Renderer::Render(const Scene& scene, const Camera& camera)
{
    Ray ray;
    ray.origin = camera.GetPosition();
    
    //  draw every pixel onto screen
    for(uint32_t y = 0; y < m_FinalRenderImage->GetHeight(); y++)
    {
        for(uint32_t x = 0; x < m_FinalRenderImage->GetWidth(); x++)
        {
            ray.direction = camera.GetRayDirections()[x + y * m_FinalRenderImage->GetWidth()];
            glm::vec4 pixelColor = TraceRay(scene, ray);
            pixelColor = glm::clamp(pixelColor, glm::vec4{0.0f}, glm::vec4{1.0f});
            
            m_RenderImageData[x + y * m_FinalRenderImage->GetWidth()] = ColorUtils::ConvertToRGBA(pixelColor);
        }
    }
    
    m_FinalRenderImage->SetData(m_RenderImageData);	//upload the image data onto GPU
}