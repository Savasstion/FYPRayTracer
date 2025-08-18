#ifndef SCENE_H
#define SCENE_H

#include <vector>

#include "Mesh.h"
#include "Sphere.h"
#include "Triangle.h"
#include "Vector3f.cuh"
#include "Vertex.h"
#include "../BaseClasses/Material.cuh"
#include "../../DataStructures/BVH.cuh"

namespace SceneSettings
{
    static Vector3f minSceneBound{-30, -30, -30};
    static Vector3f maxSceneBound{30, 30, 30};
}

struct Scene
{
    //  vectors of primitives in the scene
    //std::vector<Sphere> spheres;
    std::vector<Vertex> vertices;   //  vertices with mesh local coordinates with no transforms 
    std::vector<Vertex> worldVertices; //   vertices that has world transforms applied
    std::vector<uint32_t> triangleVertexIndices;    //  every three index of a vertex represents a triangle in this list of indices
    std::vector<Triangle> triangles;    //  used as a buffer to group triangles up for easier calculations like for ray intersection test or bvh
    std::vector<Mesh> meshes;

    //  Acceleration Structure
    BVH tlas;
    std::vector<BVH> blasOfSceneMeshes;
    
    std::vector<Material> materials;

    Mesh* AddNewMeshToScene(std::vector<Vertex>& meshVertices, std::vector<uint32_t>& meshTriangleVertexIndices,
        glm::vec3& pos, glm::vec3& rotation, glm::vec3& scale, int materialIndex);
    void UpdateSceneMeshTransform(uint32_t meshIndex, const glm::vec3& newPos, const glm::vec3& newRot, const glm::vec3& newScale);
    void UpdateAllTransformedSceneMeshes();
    std::vector<BVH::Node> CreateBVHnodesFromSceneTriangles();  //obsolete now
    std::vector<BVH::Node> CreateBVHnodesFromSceneMeshes();
};

#endif

