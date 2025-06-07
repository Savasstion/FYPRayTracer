#pragma once
#include <memory>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include "Walnut/Image.h"

class Renderer
{
private:
    std::shared_ptr<Walnut::Image> m_FinalRenderImage;
    uint32_t* m_RenderImageData = nullptr;

    glm::vec4 PerPixel(glm::vec2 pixelCoord);
public:
    Renderer() = default;
    
    void OnResize(uint32_t width, uint32_t height);
    void Render();
    std::shared_ptr<Walnut::Image> GetFinalRenderImage() const {return m_FinalRenderImage;}
};
