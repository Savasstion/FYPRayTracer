#ifndef  RENDERER_H
#define RENDERER_H

#include "Walnut/Image.h"
#include <memory>
#include <glm/glm.hpp>
#include "../BaseClasses/Camera.h"
#include "../BaseClasses/Ray.h"
#include "../BaseClasses/Scene.h"


class Renderer
{
private:
    std::shared_ptr<Walnut::Image> m_FinalRenderImage;
    uint32_t* m_RenderImageData = nullptr;
    const Scene* m_ActiveScene = nullptr;
    const Camera* m_ActiveCamera = nullptr;

    glm::vec4 PerPixel(uint32_t x, uint32_t y);    //RayGen
    RayHitPayload TraceRay(const Ray& ray);
    RayHitPayload ClosestHit(const Ray& ray, float hitDistance, int objectIndex);
    RayHitPayload Miss(const Ray& ray);
public:
    Renderer() = default;
    
    void OnResize(uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalRenderImage() const {return m_FinalRenderImage;}
};

#endif