#include "Triangle.h"

#include <glm/gtx/norm.hpp>

glm::vec3 Triangle::GetBarycentricCoords(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
{
    return (p0 + p1 + p2) / 3.0f;
}

glm::vec3 Triangle::GetTriangleNormal(const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2)
{
    // Average the vertex normals
    glm::vec3 normal = (n0 + n1 + n2) / 3.0f;
    
    return glm::normalize(normal);
}

float Triangle::GetTriangleArea(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
{
    glm::vec3 edge1 = p1 - p0;
    glm::vec3 edge2 = p2 - p0;
    return 0.5f * glm::length(glm::cross(edge1, edge2));
}

float Triangle::GetTriangleAreaSquared(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2)
{
    glm::vec3 edge1 = p1 - p0;
    glm::vec3 edge2 = p2 - p0;
    return 0.25f * glm::length2(glm::cross(edge1, edge2)); // (area^2)
}
