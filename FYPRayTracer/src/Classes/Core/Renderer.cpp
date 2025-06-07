#include "Renderer.h"
#include "Walnut/Random.h"
#include "../../Utility/ColorUtils.h"

glm::vec4 Renderer::PerPixel(glm::vec2 pixelCoord)   //  Project a ray per pixel to determine pixel output
{
    float radius = 0.5f;
    glm::vec3 rayOrigin(0.0f, 0.0f, 1.0f);  //  we take reference to OpenGL which its forward direction is z = -1
    glm::vec3 rayDirection(pixelCoord.x, pixelCoord.y, -1.0f);
    //glm::normalize(rayDirection);

    ////  Sphere Ray Hit Detection
    //  (bx^2 + by^2)t^2 + (2(axbx + ayby))t + (ax^2 + ay^2 - r^2) = 0
    //  similar to ax^2 + bx + c = 0
    //  where
    //  a = ray origin
    //  b = ray direction
    //  r = circle radius
    //  t = hit distance
    float a = glm::dot(rayDirection, rayDirection); //same as a = rayDirection.x * rayDirection.x + rayDirection.y * rayDirection.y + rayDirection.z * rayDirection.z;
    float b = 2.0f * glm::dot(rayOrigin, rayDirection);
    float c = glm::dot(rayOrigin,rayOrigin) - radius * radius;

    //  The discriminant of the quadratic formula
    //  b^2 - 4ac
    //  Used to determine if there is a ray hit to the sphere
    float discriminant = b * b - 4.0f * a * c;
    if(discriminant < 0.0f)
        return {0,0, 0, 1};  //  background color which is black

    //  Quadratic formula
    //  -b +- sqrt(discriminant) / 2a
    //float t0 = (-b + glm::sqrt(discriminant)) / (2.0f * a);
    float closestHit = (-b - glm::sqrt(discriminant)) / (2.0f * a);   //  since a will always be positive, the minus part of the formula will always be smaller so thus closer to ray origin
    glm::vec3 hitPoint = rayOrigin + rayDirection * closestHit;

    //To get our circle's normal vector onm the hitPoint, normal = hitPoint - centerOfCircle. But since our circle right now is at origin, normal = hitPoint
    glm::vec3 normal = glm::normalize(hitPoint);

    //  light direction
    glm::vec3 lightDir = glm::normalize(glm::vec3(-1,-1,-1));
    
    float d = glm::max(glm::dot(normal, -lightDir), 0.0f);  //  ==  cos(angle)
    
    glm::vec3 sphereColor(1,0,1);
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

void Renderer::Render()
{
    //  draw every pixel onto screen
    for(uint32_t y = 0; y < m_FinalRenderImage->GetHeight(); y++)
    {
        for(uint32_t x = 0; x < m_FinalRenderImage->GetWidth(); x++)
        {
            glm::vec2 pixelCoord = {(float)x / m_FinalRenderImage->GetWidth(), (float)y / m_FinalRenderImage->GetHeight()}; //  in range of 0 to 1
            pixelCoord = pixelCoord * 2.0f - 1.0f;  //  remap to range of -1 to 1
            
            glm::vec4 pixelColor = PerPixel(pixelCoord);
            pixelColor = glm::clamp(pixelColor, glm::vec4(0.0f), glm::vec4(1.0f));
            m_RenderImageData[x + y * m_FinalRenderImage->GetWidth()] = ColorUtils::ConvertToRGBA(pixelColor);
        }
    }
    
    m_FinalRenderImage->SetData(m_RenderImageData);	//upload the image data onto GPU
}