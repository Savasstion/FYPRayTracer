#ifndef CAMERA_H
#define CAMERA_H

#include <glm/glm.hpp>
#include <vector>

class Camera
{
private:
    glm::mat4 m_Projection{1.0f};
    glm::mat4 m_View{1.0f};
    glm::mat4 m_InverseProjection{1.0f};
    glm::mat4 m_InverseView{1.0f};

    float m_VerticalFOV = 45.0f;
    float m_NearClip = 0.1f;
    float m_FarClip = 100.0f;

    glm::vec3 m_Position{0.0f, 0.0f, 0.0f};
    glm::vec3 m_ForwardDirection{0.0f, 0.0f, 0.0f};

    // Cached ray directions
    std::vector<glm::vec3> m_RayDirections;

    glm::vec2 m_LastMousePosition{0.0f, 0.0f};
    uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;


    void RecalculateProjection();
    void RecalculateView();
    void RecalculateRayDirections();

public:
    Camera(float verticalFOV, float nearClip, float farClip);

    bool OnUpdate(float ts);
    void OnResize(uint32_t width, uint32_t height);
    void UpdateCameraView();

    const glm::mat4& GetProjection() const { return m_Projection; }
    const glm::mat4& GetInverseProjection() const { return m_InverseProjection; }
    const glm::mat4& GetView() const { return m_View; }
    const glm::mat4& GetInverseView() const { return m_InverseView; }

    glm::vec3& GetPosition() { return m_Position; }
    glm::vec3& GetDirection() { return m_ForwardDirection; }

    void SetPosition(const glm::vec3& m_position)
    {
        m_Position = m_position;
        UpdateCameraView();
    }

    void SetDirection(const glm::vec3& m_forward_direction)
    {
        m_ForwardDirection = m_forward_direction;
        UpdateCameraView();
    }

    const std::vector<glm::vec3>& GetRayDirections() const { return m_RayDirections; }

    float GetRotationSpeed();
    uint32_t GetViewportWidth() const { return m_ViewportWidth; }
    uint32_t GetViewportHeight() const { return m_ViewportHeight; }

private:
};

#endif
