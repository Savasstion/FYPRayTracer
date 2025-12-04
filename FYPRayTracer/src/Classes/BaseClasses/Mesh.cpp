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
    size_t indexCount = 6 * (n_stacks - 2) * n_slices + 3 * 2 * n_slices;
    outVertices.resize(vertexCount);
    outIndices.resize(indexCount);

    // --- Top pole ---
    outVertices[0] = Vertex{glm::vec3(0, radius, 0), glm::vec3(0, 1, 0), glm::vec2(0.5f, 1.0f)};
    uint32_t topIndex = 0;

    // --- Bottom pole ---
    uint32_t bottomIndex = static_cast<uint32_t>(vertexCount - 1);
    outVertices[bottomIndex] = Vertex{glm::vec3(0, -radius, 0), glm::vec3(0, -1, 0), glm::vec2(0.5f, 0.0f)};

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
            auto uv = glm::vec2(float(j) / n_slices, 1.0f - float(i + 1) / n_stacks);

            outVertices[1 + i * n_slices + j] = Vertex{pos, normal, uv};
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
    auto q = glm::quat(rotationRadians);
    glm::mat4 R = glm::toMat4(q);
    glm::mat4 S = glm::scale(glm::mat4(1.0f), mesh.scale);
    glm::mat4 T = glm::translate(glm::mat4(1.0f), mesh.position);
    mesh.worldTransformMatrix = T * R * S;
}

void Mesh::UpdateMeshAABB(Mesh& mesh, std::vector<Vertex>& vertices, std::vector<Vertex>& worldVertices,
                          std::vector<Triangle>& triangles, const std::vector<uint32_t>& triangleVertexIndices)
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
    uint32_t* outObjectOffset) const
{
    uint32_t meshTriangleStart = indexStart / 3;
    uint32_t meshTriangleCount = indexCount / 3;
    *outObjectOffset = meshTriangleStart;
    //  make sure BVH knows how much to offset to first triangle, only applicable for if BVH uses Morton Codes

    std::vector<BVH::Node> leafNodes;
    leafNodes.reserve(meshTriangleCount);

    for (uint32_t i = 0; i < meshTriangleCount; i++)
    {
        uint32_t triIndex = meshTriangleStart + i;
        const Triangle& tri = triangles[triIndex];

        // Create a leaf node for this triangle
        BVH::Node node(triIndex, tri.aabb);
        leafNodes.push_back(node);
    }

    return leafNodes;
}

std::vector<LightTree::Node> Mesh::CreateLightTreenodesFromEmmisiveMeshTriangles(
    const std::vector<Triangle>& triangles, const std::vector<Material>& materials,
    const std::vector<Vertex>& worldVertices) const
{
    uint32_t meshTriangleStart = indexStart / 3;
    uint32_t meshTriangleCount = indexCount / 3;

    std::vector<LightTree::Node> leafNodes;
    leafNodes.reserve(meshTriangleCount);

    //  only do if emissive
    if (glm::length2(materials[materialIndex].GetEmission()) > 0.0f)
        for (uint32_t i = 0; i < meshTriangleCount; i++)
        {
            uint32_t triIndex = meshTriangleStart + i;

            auto& v0 = worldVertices[triangles[triIndex].v0];
            auto& v1 = worldVertices[triangles[triIndex].v1];
            auto& v2 = worldVertices[triangles[triIndex].v2];
            constexpr float PIhalf = MathUtils::pi / 2.0f;
            float emmisiveRadiance = materials[materialIndex].GetEmissionRadiance();

            glm::vec3 baryCentricCoord = Triangle::GetBarycentricCoords(v0.position, v1.position, v2.position);
            ConeBounds bounds_o;
            bounds_o.theta_e = PIhalf;
            bounds_o.theta_o = 0.0f;
            bounds_o.axis = Triangle::GetTriangleNormal(v0.normal, v1.normal, v2.normal);
            float area = Triangle::GetTriangleArea(v0.position, v1.position, v2.position);
            float energy = area * emmisiveRadiance * MathUtils::pi;

            leafNodes.emplace_back(triIndex, baryCentricCoord, triangles[triIndex].aabb, bounds_o, energy);
        }

    return leafNodes;
}

void Mesh::ProcessMesh(const aiMesh* mesh, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    size_t baseVertex = vertices.size();

    // --- Vertices ---
    for (unsigned int i = 0; i < mesh->mNumVertices; i++)
    {
        Vertex vertex;
        vertex.position = {
            mesh->mVertices[i].x,
            mesh->mVertices[i].y,
            mesh->mVertices[i].z
        };

        if (mesh->HasNormals())
        {
            vertex.normal = {
                mesh->mNormals[i].x,
                mesh->mNormals[i].y,
                mesh->mNormals[i].z
            };
        }

        if (mesh->HasTextureCoords(0))
        {
            vertex.uv = {
                mesh->mTextureCoords[0][i].x,
                mesh->mTextureCoords[0][i].y
            };
        }

        vertices.push_back(vertex);
    }

    // --- Indices ---
    for (unsigned int i = 0; i < mesh->mNumFaces; i++)
    {
        const aiFace& face = mesh->mFaces[i];
        for (unsigned int j = 0; j < face.mNumIndices; j++)
        {
            indices.push_back(static_cast<uint32_t>(baseVertex + face.mIndices[j]));
        }
    }
}

void Mesh::ProcessNode(const aiNode* node, const aiScene* scene, std::vector<Vertex>& vertices,
                       std::vector<uint32_t>& indices)
{
    // Process all meshes in this node
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        const aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
        ProcessMesh(mesh, vertices, indices);
    }

    // Recursively process children
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        ProcessNode(node->mChildren[i], scene, vertices, indices);
    }
}

void Mesh::GenerateMesh(const std::string& filepath, std::vector<Vertex>& outVertices,
                        std::vector<uint32_t>& outIndices, bool toFlipUV)
{
    Assimp::Importer importer;

    const aiScene* scene = importer.ReadFile(
    filepath,
    aiProcess_Triangulate |
    aiProcess_GenSmoothNormals |
    aiProcess_FlipUVs |
    aiProcess_JoinIdenticalVertices |
    aiProcess_CalcTangentSpace |
    aiProcess_ConvertToLeftHanded
);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cerr << "ASSIMP Error: " << importer.GetErrorString() << std::endl;
        return;
    }

    outVertices.clear();
    outIndices.clear();

    ProcessNode(scene->mRootNode, scene, outVertices, outIndices);

    // Left-Handed Fix (for OpenGL-like systems) 
    // ASSIMP often loads in right-handed space
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(outVertices.size()); i++)
    {
        outVertices[i].position.z *= -1.0f;
        outVertices[i].normal.z *= -1.0f;
    }

    if (toFlipUV)
    {
#pragma omp parallel for
        for (int i = 0; i < static_cast<int>(outVertices.size()); i++)
        {
            outVertices[i].uv.y = 1.0f - outVertices[i].uv.y;
        }
    }
}
