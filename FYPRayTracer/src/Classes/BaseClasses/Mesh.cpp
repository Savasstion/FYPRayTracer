#include "Mesh.h"


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