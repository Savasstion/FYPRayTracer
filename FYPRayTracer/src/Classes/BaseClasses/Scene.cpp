#include "Scene.h"
#include <glm/gtx/quaternion.hpp>


Mesh* Scene::AddNewMeshToScene(std::vector<Vertex>& meshVertices,
    std::vector<uint32_t>& meshTriangleVertexIndices,
    glm::vec3& pos, glm::vec3& rotation, glm::vec3& scale,
    int materialIndex)
{
    glm::vec3 rotationRadians = glm::radians(rotation);
    glm::quat q = glm::quat(rotationRadians);
    glm::mat4 R = glm::toMat4(q);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    glm::mat4 T = glm::translate(glm::mat4(1.0f), pos);
    glm::mat4 worldMatrix = T * R * S;

    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(worldMatrix)));

    uint32_t vertexStart = static_cast<uint32_t>(vertices.size());
    uint32_t indexStart = static_cast<uint32_t>(triangleVertexIndices.size());

    Vector3f meshAABBLow(FLT_MAX, FLT_MAX, FLT_MAX);
    Vector3f meshAABBHigh(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // Transform and store each vertex
    for (const auto& v : meshVertices)
    {
        Vertex transformed;
        transformed.position = glm::vec3(worldMatrix * glm::vec4(v.position, 1.0f));
        transformed.normal = glm::normalize(normalMatrix * v.normal);
        transformed.uv = v.uv;

        vertices.push_back(transformed);
        worldVertices.push_back(transformed);

        meshAABBLow = meshAABBLow.min(transformed.position);
        meshAABBHigh = meshAABBHigh.max(transformed.position);
    }

    // Create triangles and calculate their AABBs
    for (size_t i = 0; i < meshTriangleVertexIndices.size(); i += 3)
    {
        Triangle tri;
        tri.v0 = vertexStart + meshTriangleVertexIndices[i + 0];
        tri.v1 = vertexStart + meshTriangleVertexIndices[i + 1];
        tri.v2 = vertexStart + meshTriangleVertexIndices[i + 2];
        tri.materialIndex = materialIndex;

        const Vector3f& p0 = worldVertices[tri.v0].position;
        const Vector3f& p1 = worldVertices[tri.v1].position;
        const Vector3f& p2 = worldVertices[tri.v2].position;

        Vector3f triLow = p0.min(p1).min(p2);
        Vector3f triHigh = p0.max(p1).max(p2);
        tri.aabb = AABB(triLow, triHigh);

        //  Update mesh AABB
        meshAABBLow = meshAABBLow.min(triLow);
        meshAABBHigh = meshAABBHigh.max(triHigh);

        triangles.push_back(tri);
    }

    for (uint32_t idx : meshTriangleVertexIndices)
        triangleVertexIndices.push_back(vertexStart + idx);

    Mesh mesh;
    mesh.position = pos;
    mesh.rotation = rotation;
    mesh.scale = scale;
    mesh.worldTransformMatrix = worldMatrix;
    mesh.vertexStart = vertexStart;
    mesh.vertexCount = static_cast<uint32_t>(meshVertices.size());
    mesh.indexStart = indexStart;
    mesh.indexCount = static_cast<uint32_t>(meshTriangleVertexIndices.size());
    mesh.materialIndex = materialIndex;
    mesh.aabb = AABB(meshAABBLow, meshAABBHigh);

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

// std::vector<BVH::Node> Scene::CreateBVHnodesFromSceneMeshes()
// {
// }
