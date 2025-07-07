#ifndef VERTEX_H
#define VERTEX_H
#include <glm/vec3.hpp>

struct Vertex
{
    glm::vec3 position{0,0,0};
    glm::vec3 normal{0,0,0};
    glm::vec3 uv{0,0,0};
};

#endif