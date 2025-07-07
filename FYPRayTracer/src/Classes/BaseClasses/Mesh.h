#ifndef MESH_H
#define MESH_H
#include <cstdint>

struct Mesh
{
    uint32_t vertexStart;
    uint32_t vertexCount;
    uint32_t indexStart;
    uint32_t indexCount;
    
    int materialIndex;
};

#endif