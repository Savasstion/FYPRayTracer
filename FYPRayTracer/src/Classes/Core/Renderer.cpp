#include "Renderer.h"
#include <execution>


void Renderer::OnResize(uint32_t width, uint32_t height)
{
    if (m_FinalRenderImage)
    {
        //  no resize necessary
        if (m_FinalRenderImage->GetWidth() == width && m_FinalRenderImage->GetHeight() == height)
            return;

        m_FinalRenderImage->Resize(width, height);

        //  Reset Depth Buffer
        ResizeDepthBuffers(width, height);
        //  Reset Normal Buffer
        ResizeNormalBuffers(width, height);
        // Reset ReSTIR DI reservoirs
        ResizeDIReservoirs(width, height);
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
    for (uint32_t i = 0; i < width; i++)
        m_ImageHorizontalIter[i] = i;
    for (uint32_t i = 0; i < height; i++)
        m_ImageVerticalIter[i] = i;
}
