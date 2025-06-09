#ifndef SPHERE_H
#define SPHERE_H

#include <glm/vec3.hpp>

#include "Material.h"

struct Sphere
{
    glm::vec3 position{0.0f,0.0f,0.0f};
    float radius = 0.5f;

    int materialIndex = 0;
};

#endif