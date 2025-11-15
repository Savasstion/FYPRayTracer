#ifndef CONE_BOUNDS_H
#define CONE_BOUNDS_H
#include <glm/vec3.hpp>
#include "AABB.cuh"
#include <glm/gtx/quaternion.hpp>
#include "../../Utility/MathUtils.cuh"

struct ConeBounds
{
    glm::vec3 axis{0.0f};
    float theta_o = 0.0f;
    float theta_e = 0.0f;

    __host__ __device__ __forceinline__ static ConeBounds UnionCone(ConeBounds a, ConeBounds b)
    {
        //  from the paper Importance Sampling of Many Lights with Adaptive Tree Splitting

        if (b.theta_o > a.theta_o)
        {
            //  Swap(a,b)
            ConeBounds tmp = a;
            a = b;
            b = tmp;
        }

        float theta_d = glm::acos(glm::dot(a.axis, b.axis));
        float theta_e = fmaxf(a.theta_e, b.theta_e);

        if (fminf(theta_d + b.theta_o, MathUtils::pi) <= a.theta_o)
            return {a.axis, a.theta_o, theta_e};

        float theta_o = (a.theta_o + theta_d + b.theta_o) * 0.5f;

        // full hemisphere?
        if (MathUtils::pi <= theta_o)
            return {a.axis, MathUtils::pi, theta_e};

        float theta_r = theta_o - a.theta_o;
        glm::vec3 rotAxis = glm::cross(a.axis, b.axis);

        glm::mat4 R = glm::rotate(glm::mat4(1.0f), theta_r, rotAxis);
        glm::vec3 axis = glm::normalize(glm::vec3(R * glm::vec4(a.axis, 0.0f)));

        return {axis, theta_o, theta_e};
    }

    __host__ __device__ __forceinline__ static ConeBounds FindConeThatEnvelopsAABBFromPoint(
        const AABB& aabb, glm::vec3 pointPos)
    {
        glm::vec3 centroid{0};
        centroid.x = aabb.centroidPos.x;
        centroid.y = aabb.centroidPos.y;
        centroid.z = aabb.centroidPos.z;
        //  define cone's axis
        glm::vec3 axis = glm::normalize(centroid - pointPos);

        //  define corner positions of the AABB
        glm::vec3 corners[8] = {
            {aabb.lowerBound.x, aabb.lowerBound.y, aabb.lowerBound.z},
            {aabb.lowerBound.x, aabb.lowerBound.y, aabb.upperBound.z},
            {aabb.lowerBound.x, aabb.upperBound.y, aabb.lowerBound.z},
            {aabb.lowerBound.x, aabb.upperBound.y, aabb.upperBound.z},
            {aabb.upperBound.x, aabb.lowerBound.y, aabb.lowerBound.z},
            {aabb.upperBound.x, aabb.lowerBound.y, aabb.upperBound.z},
            {aabb.upperBound.x, aabb.upperBound.y, aabb.lowerBound.z},
            {aabb.upperBound.x, aabb.upperBound.y, aabb.upperBound.z}
        };

        //   for every corner, get the biggest angle that encompasses all corners 
        float maxTheta = 0.0f;
        for (int i = 0; i < 8; i++)
        {
            glm::vec3 dir = glm::normalize(glm::vec3(corners[i] - pointPos));
            float cosTheta = glm::dot(axis, dir);
            cosTheta = glm::clamp(cosTheta, -1.0f, 1.0f); // safety against float errors
            float theta = acosf(cosTheta);
            maxTheta = fmaxf(maxTheta, theta);
        }

        //  use theta_o to store the cone's angle
        ConeBounds result;
        result.axis = axis;
        result.theta_o = maxTheta;
        result.theta_e = 0.0f;

        return result;
    }
};

#endif
