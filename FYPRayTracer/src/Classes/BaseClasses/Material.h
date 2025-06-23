#pragma once

struct Material
{
    glm::vec3 albedo{1.0f};
    float roughness = 1.0f;
    float metallic = 0.0f;
    glm::vec3 emissionColor{0.0f};
    float emissionPower = 0.0f;

    glm::vec3 GetEmission() const {return emissionColor * emissionPower;}
    float GetEmissionPower() const {return emissionPower;}
};
