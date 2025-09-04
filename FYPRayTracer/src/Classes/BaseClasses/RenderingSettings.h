#ifndef SCENE_SETTINGS_H
#define SCENE_SETTINGS_H
#include <glm/vec3.hpp>

struct RenderingSettings
{
    bool toAccumulate = true;
    int lightBounces = 1;
    int sampleCount = 1;
    glm::vec3 skyColor{0, 0, 0};
};


#endif
