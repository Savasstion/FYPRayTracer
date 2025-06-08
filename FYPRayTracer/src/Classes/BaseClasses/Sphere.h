#ifndef SPHERE_H
#define SPHERE_H

#include <glm/vec3.hpp>

struct Sphere
{
    glm::vec3 position{0.0f,0.0f,0.0f};
    float radius = 0.5f;

    glm::vec3 albedo{1.0f};
};

#endif