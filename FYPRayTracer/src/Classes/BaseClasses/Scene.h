#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "Mesh.h"
#include "Sphere.h"
#include "Texture.cuh"
#include "Triangle.cuh"
#include "Vector3f.cuh"
#include "Vertex.h"
#include "../BaseClasses/Material.cuh"
#include "../../DataStructures/BVH.cuh"
#include "../../DataStructures/LightTree.cuh"
#include "../../Enums/MaterialPropertiesEnum.h"
#include "../Managers/SceneManager.h"

namespace SceneSettings //  only needed for Morton Codes
{
    static Vector3f minSceneBound{-30, -30, -30};
    static Vector3f maxSceneBound{30, 30, 30};
}

struct Scene
{
    SceneManager sceneManager;
    
    //  vectors of primitives in the scene
    std::vector<Vertex> vertices;   //  vertices with mesh local coordinates with no transforms 
    std::vector<Vertex> worldVertices; //   vertices that has world transforms applied
    std::vector<uint32_t> triangleVertexIndices;    //  every three index of a vertex represents a triangle in this list of indices
    std::vector<Triangle> triangles;    //  used as a buffer to group triangles up for easier calculations like for ray intersection test or bvh
    std::vector<uint32_t> emissiveTriangles;
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<Texture> textures;

    //  Acceleration Structure
    BVH tlas;   //  for ray-mesh intersections
    LightTree lightTree_tlas;    //  Light Tree for light-source sampling / NEE

    
    Mesh* AddNewMeshToScene(std::vector<Vertex>& meshVertices, std::vector<uint32_t>& meshTriangleVertexIndices, const glm::vec3& pos, const glm::vec3& rotation, const glm::vec3& scale, int materialIndex);
    void UpdateSceneMeshTransform(uint32_t meshIndex, const glm::vec3& newPos, const glm::vec3& newRot, const glm::vec3& newScale);
    void UpdateAllTransformedSceneMeshes();
    std::vector<BVH::Node> CreateBVHnodesFromSceneTriangles() const;  //    obsolete now that we have a level system : TLAS/BLAS 
    std::vector<BVH::Node> CreateBVHnodesFromSceneMeshes() const;
    std::vector<LightTree::Node> CreateLightTreeNodesFromEmissiveTriangles();  
    std::vector<LightTree::Node> CreateLightTreeNodesFromBLASLightTrees() const;
    uint32_t AddNewTexture(std::string& textureFilePath, Material& mat, MaterialPropertiesEnum matProperty);
    void InitSceneEmissiveTriangles();
};

#endif

