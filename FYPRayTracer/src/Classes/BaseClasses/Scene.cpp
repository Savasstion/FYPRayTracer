#include "Scene.h"
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/norm.hpp>
#include "../Core/Renderer.h"
#include "../../../vendor/stb_image/stb_image.h"

Mesh* Scene::AddNewMeshToScene(std::vector<Vertex>& meshVertices,
                               std::vector<uint32_t>& meshTriangleVertexIndices,
                               const glm::vec3& pos,
                               const glm::vec3& rotation,
                               const glm::vec3& scale,
                               int materialIndex)
{
    Mesh mesh;
    mesh.position = pos;
    mesh.rotation = rotation;
    mesh.scale = scale;
    mesh.materialIndex = materialIndex;

    // --- offsets into scene global buffers ---
    mesh.vertexStart = static_cast<uint32_t>(vertices.size());
    mesh.vertexCount = static_cast<uint32_t>(meshVertices.size());
    mesh.indexStart = static_cast<uint32_t>(triangleVertexIndices.size());
    mesh.indexCount = static_cast<uint32_t>(meshTriangleVertexIndices.size());

    // append local vertices
    vertices.insert(vertices.end(), meshVertices.begin(), meshVertices.end());

    // build transform
    glm::mat4 translation = glm::translate(glm::mat4(1.0f), pos);
    glm::mat4 rotationMat = glm::yawPitchRoll(
        glm::radians(rotation.y),
        glm::radians(rotation.x),
        glm::radians(rotation.z));
    glm::mat4 scaling = glm::scale(glm::mat4(1.0f), scale);

    mesh.worldTransformMatrix = translation * rotationMat * scaling;

    // append world vertices (transformed)
    for (const auto& v : meshVertices)
    {
        Vertex worldV = v;
        glm::vec4 p = mesh.worldTransformMatrix * glm::vec4(v.position, 1.0f);
        glm::vec4 n = mesh.worldTransformMatrix * glm::vec4(v.normal, 0.0f);

        worldV.position = glm::vec3(p) / p.w;
        worldV.normal = glm::normalize(glm::vec3(n));
        worldVertices.push_back(worldV);
    }

    // append indices (offset by vertexStart)
    for (uint32_t idx : meshTriangleVertexIndices)
        triangleVertexIndices.push_back(mesh.vertexStart + idx);

    // generate triangles (store indices only)
    for (size_t i = 0; i < meshTriangleVertexIndices.size(); i += 3)
    {
        Triangle tri;
        tri.v0 = mesh.vertexStart + meshTriangleVertexIndices[i + 0];
        tri.v1 = mesh.vertexStart + meshTriangleVertexIndices[i + 1];
        tri.v2 = mesh.vertexStart + meshTriangleVertexIndices[i + 2];
        tri.materialIndex = materialIndex;

        // compute triangle AABB right here
        const glm::vec3& p0 = worldVertices[tri.v0].position;
        const glm::vec3& p1 = worldVertices[tri.v1].position;
        const glm::vec3& p2 = worldVertices[tri.v2].position;

        tri.aabb.lowerBound = glm::min(glm::min(p0, p1), p2);
        tri.aabb.upperBound = glm::max(glm::max(p0, p1), p2);

        tri.aabb.centroidPos = AABB::FindCentroid(tri.aabb);

        triangles.push_back(tri);
    }

    // compute mesh AABB by merging its triangles’ AABBs
    if (!triangles.empty())
    {
        AABB meshBounds = triangles.back().aabb; // start with last added
        for (size_t i = triangles.size() - (mesh.indexCount / 3); i < triangles.size(); i++)
            meshBounds = AABB::UnionAABB(meshBounds, triangles[i].aabb);
        mesh.aabb = meshBounds;
        mesh.aabb.centroidPos = AABB::FindCentroid(mesh.aabb);
    }


    meshes.push_back(mesh);
    return &meshes.back();
}

std::vector<BVH::Node> Scene::CreateBVHnodesFromSceneTriangles() const
{
    std::vector<BVH::Node> leafNodes;
    leafNodes.reserve(triangles.size());

    for (size_t i = 0; i < triangles.size(); ++i)
    {
        const Triangle& tri = triangles[i];
        const AABB& aabb = tri.aabb;

        // Create a leaf node for the triangle (objectIndex = i)
        leafNodes.emplace_back(i, aabb);
    }

    return leafNodes;
}

std::vector<BVH::Node> Scene::CreateBVHnodesFromSceneMeshes() const
{
    std::vector<BVH::Node> leafNodes;
    leafNodes.reserve(meshes.size());

    for (size_t i = 0; i < meshes.size(); i++)
    {
        const Mesh& mesh = meshes[i];
        const AABB& aabb = mesh.aabb;

        // Create a leaf node for the mesh (objectIndex = i)
        leafNodes.emplace_back(i, aabb);
    }

    return leafNodes;
}

std::vector<LightTree::Node> Scene::CreateLightTreeNodesFromEmissiveTriangles()
{
    std::vector<LightTree::Node> leafNodes;
    leafNodes.reserve(emissiveTriangles.size());

    for (uint32_t i = 0; i < emissiveTriangles.size(); i++)
    {
        Triangle& tri = triangles[emissiveTriangles[i]];
        if (glm::length2(materials[tri.materialIndex].GetEmission()) > 0.0f)
        {
            auto& v0 = worldVertices[tri.v0];
            auto& v1 = worldVertices[tri.v1];
            auto& v2 = worldVertices[tri.v2];
            constexpr float PIhalf = MathUtils::pi / 2.0f;
            float emmisiveRadiance = materials[tri.materialIndex].GetEmissionRadiance();

            uint32_t triIndex = tri.v0 / 3;
            glm::vec3 baryCentricCoord = Triangle::GetBarycentricCoords(v0.position, v1.position, v2.position);
            ConeBounds bounds_o;
            bounds_o.theta_e = PIhalf;
            bounds_o.theta_o = 0.0f;
            bounds_o.axis = Triangle::GetTriangleNormal(v0.normal, v1.normal, v2.normal);
            float area = Triangle::GetTriangleArea(v0.position, v1.position, v2.position);
            float energy = area * emmisiveRadiance * MathUtils::pi;

            leafNodes.emplace_back(triIndex, baryCentricCoord, tri.aabb, bounds_o, energy);
        }
    }

    return leafNodes;
}

std::vector<LightTree::Node> Scene::CreateLightTreeNodesFromBLASLightTrees() const
{
    std::vector<LightTree::Node> leafNodes;
    leafNodes.reserve(meshes.size() / 10);
    //  allocated a quarter's worth first before needing to increase capacity automatically

    for (uint32_t i = 0; i < meshes.size(); i++)
    {
        //  check if got valid or constructed light tree 
        if (meshes[i].lightTree_blas.nodeCount > 0)
        {
            // get parent root node and convert it to a leaf node for the TLAS
            LightTree::Node node = meshes[i].lightTree_blas.nodes[meshes[i].lightTree_blas.rootIndex];
            node.emitterIndex = i; //  emmiterIndex will now refer to which mesh's lightTree_blas this node refers to
            node.offset = 0;
            //  since no barycentric coords of a triangle, substitute with AABB centroid pos instead
            node.position.x = node.bounds_w.centroidPos.x;
            node.position.y = node.bounds_w.centroidPos.y;
            node.position.z = node.bounds_w.centroidPos.z;
            node.isLeaf = true;

            leafNodes.push_back(node);
        }
    }

    return leafNodes;
}

uint32_t Scene::AddNewTexture(std::string& textureFilePath, Material& mat, MaterialPropertiesEnum matProperty)
{
    if (!textureFilePath.empty())
        textures.emplace_back(textureFilePath);

    switch (matProperty)
    {
    case ALBEDO:
        mat.isUseAlbedoMap = true;
        break;

    case ROUGHNESS:
    case METALLIC:
        break;
    default:
        break;
    }

    return static_cast<uint32_t>(textures.size() - 1);
}

void Scene::InitSceneEmissiveTriangles()
{
    emissiveTriangles.clear();
    emissiveTriangles.reserve(triangles.size() / 10); //  arbitrarily reserve 1/10 worth of max possible space 

    for (uint32_t i = 0; i < triangles.size(); i++)
    {
        if (glm::length2(materials[triangles[i].materialIndex].GetEmission()) > 0.0f)
        {
            emissiveTriangles.push_back(i);
        }
    }
}

void Scene::CreateNewMaterialInScene()
{
    materials.emplace_back();
}

void Scene::CreateNewTextureInScene(std::string& imageFilePath)
{
    int w, h, channels;
    stbi_uc* data = stbi_load(imageFilePath.c_str(), &w, &h, &channels, 4);
    if (!data)
    {
        std::cerr << ("Failed to load image: " + imageFilePath) << std::endl;
        stbi_image_free(data);
        return;
    }
    textures.emplace_back(imageFilePath);
}

void Scene::CreateNewMeshInScene(std::string& meshFilePath)
{
    std::vector<Vertex> meshVertices;
    std::vector<uint32_t> meshIndices;
    Mesh::GenerateMesh(meshFilePath, meshVertices, meshIndices, false);

    //	Set transforms
    glm::vec3 pos{0, 0, 0};
    glm::vec3 rot{0, 0, 0};
    glm::vec3 scale{1, 1, 1};

    //	Init mesh into scene
    Mesh* meshPtr = AddNewMeshToScene(meshVertices,
                                              meshIndices,
                                              pos,
                                              rot,
                                              scale,
                                              0);
            
    //	Build BVH for ray collision for mesh
    uint32_t triOffset = 0;
    auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(triangles, &triOffset);
    meshPtr->blas.objectOffset = triOffset;
    meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

    //	Build Light Tree for Light Source Sampling for mesh
    auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(
        triangles, materials, worldVertices);
    if (lightTreeEmitterNodes.empty())
        meshPtr->lightTree_blas.nodeCount = 0;
    else
        meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                   static_cast<uint32_t>(lightTreeEmitterNodes.size()));

    //	Scene TLAS Construction
    auto tlasObjectNodes = CreateBVHnodesFromSceneMeshes();
    tlas.ConstructBVH_SAH(tlasObjectNodes.data(), tlasObjectNodes.size());

    //	Init Scene Emissive Light Source list
    InitSceneEmissiveTriangles();

    //	Scene Light Tree TLAS Construction
    auto tlasLightTreeEmitterNodes = CreateLightTreeNodesFromBLASLightTrees();
    lightTree_tlas.ConstructLightTree(tlasLightTreeEmitterNodes.data(),
                                              static_cast<uint32_t>(tlasLightTreeEmitterNodes.size()));
}
