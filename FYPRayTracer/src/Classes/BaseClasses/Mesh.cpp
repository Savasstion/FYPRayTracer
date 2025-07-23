#include "Mesh.h"
#include <glm/detail/type_quat.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>



void Mesh::GenerateSphereMesh(float radius, int latitudeSegments, int longitudeSegments, std::vector<Vertex>& outVertices, std::vector<uint32_t>& outIndices)
{
    outVertices.clear();
    outIndices.clear();

    // Generate vertices
    for (int lat = 0; lat <= latitudeSegments; lat++)
    {
        float theta = lat * MathUtils::pi / latitudeSegments;
        float sinTheta = sin(theta);
        float cosTheta = cos(theta);

        for (int lon = 0; lon <= longitudeSegments; lon++)
        {
            float phi = lon * 2.0f * MathUtils::pi / longitudeSegments;
            float sinPhi = sin(phi);
            float cosPhi = cos(phi);

            glm::vec3 position;
            position.x = radius * sinTheta * cosPhi;
            position.y = radius * cosTheta;
            position.z = radius * sinTheta * sinPhi;

            glm::vec3 normal = glm::normalize(position);  // outward-pointing normal for a sphere
            glm::vec2 uv;
            uv.x = static_cast<float>(lon) / static_cast<float>(longitudeSegments);
            uv.y = static_cast<float>(lat) / static_cast<float>(latitudeSegments);

            outVertices.push_back(Vertex{ position, normal, uv });
        }
    }

    // Generate triangle indices
    for (int lat = 0; lat < latitudeSegments; ++lat)
    {
        for (int lon = 0; lon < longitudeSegments; ++lon)
        {
            int first = (lat * (longitudeSegments + 1)) + lon;
            int second = first + longitudeSegments + 1;

            // Triangle 1
            outIndices.push_back(first);
            outIndices.push_back(second);
            outIndices.push_back(first + 1);

            // Triangle 2
            outIndices.push_back(second);
            outIndices.push_back(second + 1);
            outIndices.push_back(first + 1);
        }
    }
}

void Mesh::UpdateWorldTransform(Mesh& mesh)
{
    glm::vec3 rotationRadians = glm::radians(mesh.rotation);
    glm::quat q = glm::quat(rotationRadians);
    glm::mat4 R = glm::toMat4(q);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), mesh.scale);
    glm::mat4 T = glm::translate(glm::mat4(1.0f), mesh.position);
    mesh.worldTransformMatrix = T * R * S;
}

void Mesh::UpdateMeshAABB(Mesh& mesh, std::vector<Vertex>& vertices, std::vector<Vertex>& worldVertices, std::vector<Triangle>& triangles, const std::vector<uint32_t>& triangleVertexIndices)
{
    glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(mesh.worldTransformMatrix)));

    // Transform vertices
    for (uint32_t i = 0; i < mesh.vertexCount; ++i)
    {
        const Vertex& localV = vertices[mesh.vertexStart + i];
        Vertex& worldV = worldVertices[mesh.vertexStart + i];

        worldV.position = glm::vec3(mesh.worldTransformMatrix * glm::vec4(localV.position, 1.0f));
        worldV.normal = glm::normalize(normalMatrix * localV.normal);
        worldV.uv = localV.uv;
    }

    // Recompute AABBs
    Vector3f meshAABBLow(FLT_MAX, FLT_MAX, FLT_MAX);
    Vector3f meshAABBHigh(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    uint32_t triStart = mesh.indexStart / 3;
    uint32_t triEnd = triStart + mesh.indexCount / 3;

    for (uint32_t i = triStart; i < triEnd; ++i)
    {
        Triangle& tri = triangles[i];
        const Vector3f& p0 = worldVertices[tri.v0].position;
        const Vector3f& p1 = worldVertices[tri.v1].position;
        const Vector3f& p2 = worldVertices[tri.v2].position;

        Vector3f triMin = p0.min(p1).min(p2);
        Vector3f triMax = p0.max(p1).max(p2);
        tri.aabb = AABB(triMin, triMax);

        meshAABBLow = meshAABBLow.min(triMin);
        meshAABBHigh = meshAABBHigh.max(triMax);
    }

    mesh.aabb = AABB(meshAABBLow, meshAABBHigh);
}
