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

    glm::vec4 TraceRay(const Scene& scene, const Ray& ray);
public:
    Renderer() = default;
    
    void OnResize(uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalRenderImage() const {return m_FinalRenderImage;}
};

#endif