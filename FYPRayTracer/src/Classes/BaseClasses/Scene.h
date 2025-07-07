#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "Sphere.h"
#include "Vertex.h"
#include "../BaseClasses/Material.h"

struct Scene
{
    //  vectors of primitives in the scene
    std::vector<Sphere> spheres;
    std::vector<Vertex> vertices;
    std::vector<Vertex> transformedVertices;
    std::vector<uint32_t> triangleIndices;  //  stores the vertexIDs of each triangle
    
    std::vector<Material> materials;
};

#endif

