#include "Scene.h"
#include <glm/fwd.hpp>
#include <glm/detail/type_quat.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

void Scene::AddNewMeshToScene(std::vector<Vertex>& meshVertices,
    std::vector<uint32_t>& meshTriangleVertexIndices,
    glm::vec3& pos, glm::vec3& rotation, glm::vec3& scale,
    int materialIndex)
{
    // Step 1: Build world transform matrix (T * R * S)
    glm::vec3 rotationRadians = glm::radians(rotation);
    glm::quat q = glm::quat(rotationRadians);
    glm::mat4 R = glm::toMat4(q);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), scale);
    glm::mat4 T = glm::translate(glm::mat4(1.0f), pos);
    glm::mat4 worldMatrix = T * R * S;

    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(worldMatrix)));

    // Step 2: Record vertex and index starting offsets
    uint32_t vertexStart = static_cast<uint32_t>(vertices.size());
    uint32_t indexStart = static_cast<uint32_t>(triangleVertexIndices.size());

    // Step 3: Transform and store each vertex into both 'vertices' and 'worldVertices'
    for (const auto& v : meshVertices)
    {
        Vertex transformed;
        transformed.position = glm::vec3(worldMatrix * glm::vec4(v.position, 1.0f));
        transformed.normal = glm::normalize(normalMatrix * v.normal);
        transformed.uv = v.uv;

        vertices.push_back(transformed);        // Optional: keep for original mesh data
        worldVertices.push_back(transformed);   // Required: used in TraceRay()
    }

    // Step 4: Create triangles using vertex indices
    for (size_t i = 0; i < meshTriangleVertexIndices.size(); i += 3)
    {
        Triangle tri;
        tri.v0 = vertexStart + meshTriangleVertexIndices[i + 0];
        tri.v1 = vertexStart + meshTriangleVertexIndices[i + 1];
        tri.v2 = vertexStart + meshTriangleVertexIndices[i + 2];
        tri.materialIndex = materialIndex;

        triangles.push_back(tri);
    }

    // Step 5: Store adjusted triangle vertex indices for indexing
    for (uint32_t idx : meshTriangleVertexIndices)
        triangleVertexIndices.push_back(vertexStart + idx);

    // Step 6: Store mesh metadata
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

    meshes.push_back(mesh);
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

        // Rebuild transform
        glm::vec3 rotationRadians = glm::radians(mesh.rotation);
        glm::quat q = glm::quat(rotationRadians);
        glm::mat4 R = glm::toMat4(q);
        glm::mat4 S = glm::scale(glm::mat4(1.0f), mesh.scale);
        glm::mat4 T = glm::translate(glm::mat4(1.0f), mesh.position);
        mesh.worldTransformMatrix = T * R * S;

        // Re-apply to worldVertices
        for (uint32_t i = 0; i < mesh.vertexCount; ++i)
        {
            const Vertex& localV = vertices[mesh.vertexStart + i];
            Vertex& worldV = worldVertices[mesh.vertexStart + i];

            worldV.position = glm::vec3(mesh.worldTransformMatrix * glm::vec4(localV.position, 1.0f));
            worldV.normal   = glm::normalize(glm::vec3(mesh.worldTransformMatrix * glm::vec4(localV.normal, 0.0f)));
            worldV.uv       = localV.uv;
        }

        // Mark as clean
        mesh.isTransformed = false;
    }
}
