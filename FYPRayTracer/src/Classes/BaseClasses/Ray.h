#ifndef RAY_H
#define RAY_H

#include <glm/vec3.hpp>


struct Ray
{
    glm::vec3 origin;
    glm::vec3 direction;
};

struct RayHitPayload
{
    float hitDistance;
    glm::vec3 worldPosition;
    glm::vec3 worldNormal;
    float u = 0.0f, v = 0.0f;

    int objectIndex;
    //  may also need in the future, the type of object (ie : Sphere, Quad or Triangle)
};

#endif
