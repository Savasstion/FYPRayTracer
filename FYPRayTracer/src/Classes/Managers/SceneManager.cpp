#include "SceneManager.h"
#include "../BaseClasses/Mesh.h"
#include "../BaseClasses/Scene.h"
#include "../Core/Renderer.h"

void SceneManager::PerformAllSceneUpdates(Scene& scene, Renderer& renderer)
{
    bool toUpdateSceneTLAS = false;

    //  do all material updates
    if(!materialsToUpdate.empty())
    {
        //  values are already set in UI
        renderer.SetSceneToBeUpdatedFlag(true);
        //  Its possible user has change the emission properties and Light Trees need to be rebuilt accordingly
        toUpdateSceneTLAS = true;
    }
    
    //  do all mesh updates here
    for(uint32_t i = 0; i < meshesToUpdate.size(); i++)
    {
        Mesh& mesh = scene.meshes[meshesToUpdate[i].meshIndex];
        bool toUpdateMeshBLASs = false;
        
        if(meshesToUpdate[i].meshTransformToBeUpdated)
        {
            Mesh::UpdateWorldTransform(mesh);

            // modify world vertices
            #pragma omp parallel for
            for (uint32_t i = 0; i < mesh.vertexCount; i++)
            {
                Vertex& worldV = scene.worldVertices[mesh.vertexStart + i];
                Vertex& meshV = scene.vertices[mesh.vertexStart + i];
                
                glm::vec4 p = mesh.worldTransformMatrix * glm::vec4(meshV.position, 1.0f);
                glm::vec4 n = mesh.worldTransformMatrix * glm::vec4(meshV.normal, 0.0f);

                worldV.position = glm::vec3(p) / p.w;
                worldV.normal = glm::normalize(glm::vec3(n));
            }

            int32_t triangleStart = mesh.indexStart / 3;
            uint32_t triangleEnd = mesh.indexStart / 3 + mesh.indexCount / 3;
            
            // modify triangles
            for (uint32_t i = triangleStart; i < triangleEnd; i++)
            {
                Triangle& tri = scene.triangles[i];

                // compute triangle AABB right here
                const glm::vec3& p0 = scene.worldVertices[tri.v0].position;
                const glm::vec3& p1 = scene.worldVertices[tri.v1].position;
                const glm::vec3& p2 = scene.worldVertices[tri.v2].position;

                tri.aabb.lowerBound = glm::min(glm::min(p0, p1), p2);
                tri.aabb.upperBound = glm::max(glm::max(p0, p1), p2);

                tri.aabb.centroidPos = AABB::FindCentroid(tri.aabb);
            }
            
            

            toUpdateSceneTLAS = true;
            renderer.SetSceneToBeUpdatedFlag(true);
            toUpdateMeshBLASs = true;
        }

        if(meshesToUpdate[i].meshMatToBeUpdated)
        {
            int32_t triangleStart = mesh.indexStart / 3;
            uint32_t triangleEnd = mesh.indexStart / 3 + mesh.indexCount / 3;

            // modify triangles' material index
            for (uint32_t i = triangleStart; i < triangleEnd; i++)
            {
                Triangle& tri = scene.triangles[i];
                tri.materialIndex = mesh.materialIndex;
            }
            
            renderer.SetSceneToBeUpdatedFlag(true);
            //  New emissive triangles could have been made
            toUpdateSceneTLAS = true;
            toUpdateMeshBLASs = true;
        }

        if(toUpdateMeshBLASs)
        {
            // compute mesh AABB by merging its triangles’ AABBs
            AABB meshBounds{};
            int32_t triangleStart = mesh.indexStart / 3;
            uint32_t triangleEnd = mesh.indexStart / 3 + mesh.indexCount / 3;
            for (uint32_t i = triangleStart; i < triangleEnd; i++)
                meshBounds = AABB::UnionAABB(meshBounds, scene.triangles[i].aabb);
            mesh.aabb = meshBounds;
            mesh.aabb.centroidPos = AABB::FindCentroid(mesh.aabb);

            //	Build BVH for ray collision
            uint32_t triOffset = 0;
            auto blasObjectNodes = mesh.CreateBVHnodesFromMeshTriangles(scene.triangles, &triOffset);
            mesh.blas.objectOffset = triOffset;
            mesh.blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

            //	Build Light Tree for Light Source Sampling
            auto lightTreeEmitterNodes = mesh.CreateLightTreenodesFromEmmisiveMeshTriangles(
                scene.triangles, scene.materials, scene.worldVertices);
            if (lightTreeEmitterNodes.empty())
                mesh.lightTree_blas.nodeCount = 0;
            else
                mesh.lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                           static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        }
    }
    
    if(toUpdateSceneTLAS)
    {
        //	Scene TLAS Construction
        auto tlasObjectNodes = scene.CreateBVHnodesFromSceneMeshes();
        scene.tlas.ConstructBVH_SAH(tlasObjectNodes.data(), tlasObjectNodes.size());
        
        //	Scene Light Tree TLAS Construction
        auto lightTreeEmitterNodes = scene.CreateLightTreeNodesFromBLASLightTrees();
        scene.lightTree_tlas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                  static_cast<uint32_t>(lightTreeEmitterNodes.size()));
    }
    
    //  clear update queue
    meshesToUpdate.clear();
    materialsToUpdate.clear();
}
