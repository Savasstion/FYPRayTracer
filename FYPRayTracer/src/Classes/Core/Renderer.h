#ifndef RENDERER_H
#define RENDERER_H

#include "Walnut/Image.h"
#include <memory>
#include <vector>
#include "../../Enums/SamplingTechniqueEnum.h"
#include "../BaseClasses/RenderingSettings.h"
#include "../BaseClasses/Scene.h"
#include "../BaseClasses/Camera.h"

class Renderer
{
private:
    RenderingSettings m_Settings;
    std::shared_ptr<Walnut::Image> m_FinalRenderImage;
    uint32_t* m_RenderImageData = nullptr;
    glm::vec4* m_AccumulationData = nullptr;
    const Scene* m_ActiveScene = nullptr;
    const Camera* m_ActiveCamera = nullptr;
    uint32_t m_FrameIndex = 1;
    bool isSceneUpdated = true;

    std::vector<uint32_t> m_ImageHorizontalIter, m_ImageVerticalIter;

public:
    Renderer() = default;
    void OnResize(uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);

    std::shared_ptr<Walnut::Image> GetFinalRenderImage() const { return m_FinalRenderImage; }
    void ResetFrameIndex() { m_FrameIndex = 1; }
    RenderingSettings& GetSettings() { return m_Settings; }
    uint32_t GetCurrentFrameIndex() const { return m_FrameIndex; }
    uint32_t* GetRenderImageDataPtr() const { return m_RenderImageData; }
};

#endif
