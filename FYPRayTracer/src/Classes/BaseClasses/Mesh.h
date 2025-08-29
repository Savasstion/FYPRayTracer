#ifndef MESH_H
#define MESH_H
#include <cstdint>
#include <glm/ext/matrix_float4x4.hpp>
#include <vector>
#include "AABB.cuh"
#include "Vertex.h"
#include "Triangle.h"
#include "../../DataStructures/BVH.cuh"


struct Mesh
{
    //  If not using ECS, an entity with a transform component, store transform data here
    //  If we are gonna create a ECS game engine, use the concept of composition and move the transform data away from here
    glm::vec3 position{0,0,0};
    glm::vec3 rotation{0,0,0};
    glm::vec3 scale{1,1,1};
    glm::mat4 worldTransformMatrix;
    bool isTransformed = false; //  if any of the transform data is modified, this should be true and then only need to do calculations to retransform the object 
    
    uint32_t vertexStart;
    uint32_t vertexCount;
    uint32_t indexStart;
    uint32_t indexCount;
    
    int materialIndex;

    AABB aabb;
    BVH blas;

    static void GenerateSphereMesh(float radius, int n_stacks, int n_slices, std::vector<Vertex>& outVertices, std::vector<uint32_t>& outIndices);
    static void UpdateWorldTransform(Mesh& mesh);
    static void UpdateMeshAABB(Mesh& mesh, std::vector<Vertex>& vertices, std::vector<Vertex>& worldVertices, 
        std::vector<Triangle>& triangles, const std::vector<uint32_t>& triangleVertexIndices);
    static std::vector<BVH::Node> CreateBVHnodesFromMeshTriangles(
        const std::vector<Triangle>& triangles,
        size_t meshTriangleStart,
        size_t meshTriangleCount,
        size_t* outObjectOffset);
};

#endif