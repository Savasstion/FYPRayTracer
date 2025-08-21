#include "Mesh.h"
#include <glm/detail/type_quat.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>



void Mesh::GenerateSphereMesh(float radius, int n_stacks, int n_slices,
                              std::vector<Vertex>& outVertices, std::vector<uint32_t>& outIndices)
{
    outVertices.clear();
    outIndices.clear();

    // Preallocate
    size_t vertexCount = 2 + (n_stacks - 1) * n_slices;
    size_t indexCount  = 6 * (n_stacks - 2) * n_slices + 3 * 2 * n_slices;
    outVertices.resize(vertexCount);
    outIndices.resize(indexCount);

    // --- Top pole ---
    outVertices[0] = Vertex{ glm::vec3(0, radius, 0), glm::vec3(0, 1, 0), glm::vec2(0.5f, 1.0f) };
    uint32_t topIndex = 0;

    // --- Bottom pole ---
    uint32_t bottomIndex = static_cast<uint32_t>(vertexCount - 1);
    outVertices[bottomIndex] = Vertex{ glm::vec3(0, -radius, 0), glm::vec3(0, -1, 0), glm::vec2(0.5f, 0.0f) };

    // --- Generate ring vertices (parallelized outer loop) ---
    #pragma omp parallel for
    for (int i = 0; i < n_stacks - 1; ++i)
    {
        float phi = MathUtils::pi * float(i + 1) / float(n_stacks);
        float y = std::cos(phi);
        float r = std::sin(phi);

        for (int j = 0; j < n_slices; ++j)
        {
            float theta = 2.0f * MathUtils::pi * float(j) / float(n_slices);
            float x = r * std::cos(theta);
            float z = r * std::sin(theta);

            glm::vec3 pos = glm::vec3(x, y, z) * radius;
            glm::vec3 normal = glm::normalize(pos);
            glm::vec2 uv = glm::vec2(float(j) / n_slices, 1.0f - float(i + 1) / n_stacks);

            outVertices[1 + i * n_slices + j] = Vertex{ pos, normal, uv };
        }
    }

    // --- Fill indices (can also parallelize by rings) ---
    uint32_t idx = 0;

    // Top cap
    for (int j = 0; j < n_slices; ++j)
    {
        uint32_t i0 = j + 1;
        uint32_t i1 = (j + 1) % n_slices + 1;
        outIndices[idx++] = topIndex;
        outIndices[idx++] = i1;
        outIndices[idx++] = i0;
    }

    // Middle quads
    for (int i = 0; i < n_stacks - 2; ++i)
    {
        int ringStart0 = 1 + i * n_slices;
        int ringStart1 = 1 + (i + 1) * n_slices;

        for (int j = 0; j < n_slices; ++j)
        {
            uint32_t i0 = ringStart0 + j;
            uint32_t i1 = ringStart0 + (j + 1) % n_slices;
            uint32_t i2 = ringStart1 + (j + 1) % n_slices;
            uint32_t i3 = ringStart1 + j;

            outIndices[idx++] = i0;
            outIndices[idx++] = i1;
            outIndices[idx++] = i2;

            outIndices[idx++] = i0;
            outIndices[idx++] = i2;
            outIndices[idx++] = i3;
        }
    }

    // Bottom cap
    int baseIndex = 1 + (n_stacks - 2) * n_slices;
    for (int j = 0; j < n_slices; ++j)
    {
        uint32_t i0 = baseIndex + j;
        uint32_t i1 = baseIndex + (j + 1) % n_slices;
        outIndices[idx++] = bottomIndex;
        outIndices[idx++] = i0;
        outIndices[idx++] = i1;
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

std::vector<BVH::Node> Mesh::CreateBVHnodesFromMeshTriangles(
    const std::vector<Triangle>& triangles,
    size_t meshTriangleStart,
    size_t meshTriangleCount,
    size_t* outObjectOffset)
{
    *outObjectOffset = meshTriangleStart;   //  make sure BVH knows how much to offset to first triangle
    
    std::vector<BVH::Node> leafNodes;
    leafNodes.reserve(meshTriangleCount);

    for (size_t i = 0; i < meshTriangleCount; i++)
    {
        size_t triIndex = meshTriangleStart + i;
        const Triangle& tri = triangles[triIndex];

        // Create a leaf node for this triangle
        BVH::Node node(triIndex, tri.aabb);
        leafNodes.push_back(node);
        
    }

    return leafNodes;
}
