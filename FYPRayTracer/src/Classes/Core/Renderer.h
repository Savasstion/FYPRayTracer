#ifndef  RENDERER_H
#define RENDERER_H

#include "Walnut/Image.h"
#include <memory>
#include <glm/glm.hpp>
#include "../BaseClasses/Camera.h"
#include "../BaseClasses/Ray.h"
#include "../BaseClasses/Scene.h"

struct Settings
{
    bool toAccumulate = true;
    int lightBounces = 1;
    int sampleCount = 1;
    glm::vec3 skyColor{1,1,1};//{0.6f, 0.7f, 0.9f};
};

class Renderer
{
private:
    Settings m_Settings;
    std::shared_ptr<Walnut::Image> m_FinalRenderImage;
    uint32_t* m_RenderImageData = nullptr;
    glm::vec4* m_AccumulationData = nullptr;
    const Scene* m_ActiveScene = nullptr;
    const Camera* m_ActiveCamera = nullptr;
    uint32_t m_FrameIndex = 1;

    std::vector<uint32_t> m_ImageHorizontalIter, m_ImageVerticalIter;

    glm::vec4 PerPixel(const uint32_t& x, const uint32_t& y, const uint8_t& maxBounces, const uint8_t& sampleCount);    //RayGen
    RayHitPayload TraceRay(const Ray& ray);
    RayHitPayload ClosestHit(const Ray& ray, float hitDistance, int objectIndex, float u, float v);
    RayHitPayload Miss(const Ray& ray);
    glm::vec3 CalculateBRDF(
    const glm::vec3& N, const glm::vec3& V, const glm::vec3& L,
    const glm::vec3& albedo, float metallic, float roughness);

public:
    Renderer() = default;
    
    void OnResize(uint32_t width, uint32_t height);
    void Render(const Scene& scene, const Camera& camera);
    std::shared_ptr<Walnut::Image> GetFinalRenderImage() const {return m_FinalRenderImage;}
    void ResetFrameIndex(){ m_FrameIndex = 1; }
    Settings& GetSettings(){ return m_Settings; }
    uint32_t GetCurrentFrameIndex() const {return m_FrameIndex;}
    uint32_t* GetRenderImageDataPtr() const {return m_RenderImageData;}
    
};

#endif