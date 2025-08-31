#include "Scene.h"
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/norm.hpp>

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
    mesh.scale    = scale;
    mesh.materialIndex = materialIndex;

    // --- offsets into scene global buffers ---
    mesh.vertexStart = static_cast<uint32_t>(vertices.size());
    mesh.vertexCount = static_cast<uint32_t>(meshVertices.size());
    mesh.indexStart  = static_cast<uint32_t>(triangleVertexIndices.size());
    mesh.indexCount  = static_cast<uint32_t>(meshTriangleVertexIndices.size());

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
    mesh.isTransformed = true;

    // append world vertices (transformed)
    for (const auto& v : meshVertices)
    {
        Vertex worldV = v;
        glm::vec4 p = mesh.worldTransformMatrix * glm::vec4(v.position, 1.0f);
        glm::vec4 n = mesh.worldTransformMatrix * glm::vec4(v.normal, 0.0f);

        worldV.position = glm::vec3(p) / p.w;
        worldV.normal   = glm::normalize(glm::vec3(n));
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



void Scene::UpdateSceneMeshTransform(uint32_t meshIndex, const glm::vec3& newPos, const glm::vec3& newRot,
    const glm::vec3& newScale)
{
    if (meshIndex >= meshes.size()) return;

    Mesh& mesh = meshes[meshIndex];
    mesh.position = newPos;
    mesh.rotation = newRot;
    mesh.scale    = newScale;
    mesh.isTransformed = true; // mark for update
}

void Scene::UpdateAllTransformedSceneMeshes()
{
    for (Mesh& mesh : meshes)
    {
        if (!mesh.isTransformed) continue;

        Mesh::UpdateWorldTransform(mesh);

        // Re-apply to worldVertices
        for (uint32_t i = 0; i < mesh.vertexCount; ++i)
        {
            const Vertex& localV = vertices[mesh.vertexStart + i];
            Vertex& worldV = worldVertices[mesh.vertexStart + i];

            worldV.position = glm::vec3(mesh.worldTransformMatrix * glm::vec4(localV.position, 1.0f));
            worldV.normal   = glm::normalize(glm::vec3(mesh.worldTransformMatrix * glm::vec4(localV.normal, 0.0f)));
            worldV.uv       = localV.uv;
        }

        Mesh::UpdateMeshAABB(mesh, vertices, worldVertices, triangles, triangleVertexIndices);
      

        // Mark as clean
        mesh.isTransformed = false;
    }
}

std::vector<BVH::Node> Scene::CreateBVHnodesFromSceneTriangles()
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

std::vector<BVH::Node> Scene::CreateBVHnodesFromSceneMeshes()
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

std::vector<LightTree::Node> Scene::CreateLightTreeNodesFromEmmisiveTriangles()
{
    std::vector<LightTree::Node> leafNodes;
    leafNodes.reserve(triangles.size() / 4);    //  allocated a quarter's worth first before needing to increase capacity automatically
    
    for(uint32_t i = 0; i < triangles.size(); i++)
    {

        if(glm::length2(materials[triangles[i].materialIndex].GetEmission()) > 0.0f)
        {
            auto& v0 = vertices[triangles[i].v0];
            auto& v1 = vertices[triangles[i].v1];
            auto& v2 = vertices[triangles[i].v2];
            constexpr float PIhalf = MathUtils::pi / 2.0f;
            float emmisiveRadiance = materials[triangles[i].materialIndex].GetEmissionRadiance();
            
            uint32_t triIndex = triangles[i].v0 / 3;
            glm::vec3 baryCentricCoord = Triangle::GetBarycentricCoords(v0.position, v1.position, v2.position);
            ConeBounds bounds_o;
            bounds_o.theta_e = PIhalf;
            bounds_o.theta_o = 0.0f;
            bounds_o.axis = Triangle::GetTriangleNormal(v0.normal, v1.normal, v2.normal);
            float area = Triangle::GetTriangleArea(v0.position, v1.position, v2.position);
            float energy = area * emmisiveRadiance * MathUtils::pi;
            
            leafNodes.emplace_back(triIndex, baryCentricCoord, triangles[i].aabb, bounds_o, energy);
        }
    }

    return leafNodes;
}

