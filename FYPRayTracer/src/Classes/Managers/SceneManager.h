#ifndef SCENE_MANAGER_H
#define SCENE_MANAGER_H
#include <vector>

struct Scene;
class Renderer;
class SceneManager
{
public:
    struct MeshUpdateParam
    {
        MeshUpdateParam(bool mesh_transform_to_be_updated, bool mesh_mat_to_be_updated, uint32_t mesh_index)
            : meshTransformToBeUpdated(mesh_transform_to_be_updated),
              meshMatToBeUpdated(mesh_mat_to_be_updated),
              meshIndex(mesh_index)
        {
        }

        MeshUpdateParam() = default;

        bool meshTransformToBeUpdated = false, meshMatToBeUpdated = false;
        uint32_t meshIndex = -1;
    };

    std::vector<MeshUpdateParam> meshesToUpdate{20}; //  stores all indices of scene meshes to be updated
    std::vector<uint32_t> materialsToUpdate{20}; //  stores all indices of scene materials to be updated


    void PerformAllSceneUpdates(Scene& scene, Renderer& renderer);
};


#endif
