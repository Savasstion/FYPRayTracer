#include <array>
#include <filesystem>

#include "Classes/BaseClasses/Camera.h"
#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Walnut/Image.h"
#include "Walnut/Timer.h"
#include "Classes/Core/Renderer.h"
#include <glm/gtc/type_ptr.hpp>
#include "Utility/MisUtils.h"
#include "Utility/tinyfiledialogs.h"

class MainLayer : public Walnut::Layer
{
private:
    uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
    Renderer m_Renderer;
    float m_CurrentFrameTime = 0.0f;
    float m_AverageFrameTime = 0.0f;
    float m_TimeToRender = 120.0f;
    float m_RenderTime = 0.0f;
    Camera m_Camera;
    Scene m_Scene;

    bool stopRender = false;

    static constexpr std::array<const char*, SamplingTechniqueEnum_COUNT> samplingTechniqueNames = {
        "Brute Force",
        "Uniform Sampling",
        "Cosine-Weighted Sampling",
        "GGX Sampling",
        "BRDF Sampling",
        "Light-Source Sampling",
        "Next Event Estimation",
        "ReSTIR DI",
        "ReSTIR GI"
    };

public:
    MainLayer()
        : m_Camera(45.0f, 0.1f, 100.0f)
    {
        Material& matMagenta = m_Scene.materials.emplace_back();
        matMagenta.albedo = {1.0f, 0.0f, 1.0f};
        matMagenta.roughness = 1.f;
        matMagenta.metallic = 0.0f;

        Material& matBlueSphere = m_Scene.materials.emplace_back();
        matBlueSphere.albedo = {0.2f, 0.3f, 1.0f};
        matBlueSphere.roughness = 0.75f;
        matBlueSphere.metallic = 0.2f;

        Material& matWhiteEmissive = m_Scene.materials.emplace_back();
        matWhiteEmissive.albedo = {1, 1, 1};
        matWhiteEmissive.emissionColor = matWhiteEmissive.albedo;
        matWhiteEmissive.emissionPower = 40.0f;

        Material& matRed = m_Scene.materials.emplace_back();
        matRed.albedo = {1.0f, 0.0f, 0.0f};
        matRed.roughness = 1.0f;
        matRed.metallic = 0.0f;

        Material& matGreen = m_Scene.materials.emplace_back();
        matGreen.albedo = {0.0f, 1.0f, 0.0f};
        matGreen.roughness = 1.0f;
        matGreen.metallic = 0.0f;

        Material& matWhite = m_Scene.materials.emplace_back();
        matWhite.albedo = {1, 1, 1};
        matWhite.roughness = 1.0f;
        matWhite.metallic = 0.0f;

        Material& matBlue = m_Scene.materials.emplace_back();
        matBlue.albedo = {0.0f, 0.0f, 1.0f};
        matBlue.roughness = 1.0f;
        matBlue.metallic = 0.0f;

        Material& matBanana = m_Scene.materials.emplace_back();
        std::string filePath = "Assets/3D Models/Test/bananaDiffuse.png";
        matBanana.albedoMapIndex = m_Scene.AddNewTexture(filePath, matBanana, ALBEDO);
        matBanana.roughness = 1.0f;
        matBanana.metallic = 0.0f;

        ////	Place Spehres
        // for(int i = -10; i < 10 ; i++)
        // {
        // 	std::vector<Vertex> sphereVertices;
        // 	std::vector<uint32_t> sphereIndices;
        // 	Mesh::GenerateSphereMesh(1, 20,20, sphereVertices, sphereIndices);
        //
        // 	//	Set transforms
        // 	glm::vec3 pos{i,i, i};
        // 	glm::vec3 rot{0,0,0};
        // 	glm::vec3 scale{1,1,1};
        //
        // 	//	Init mesh into scene
        // 	Mesh* meshPtr = m_Scene.AddNewMeshToScene(sphereVertices,
        // 					sphereIndices,
        // 					pos,
        // 					rot,
        // 					scale,
        // 					1);
        //
        // 	//	Build BVH for ray collision
        // 	uint32_t triOffset = 0;
        // 	auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, &triOffset);
        // 	meshPtr->blas.objectOffset = triOffset;
        // 	meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());
        //
        // 	//	Build Light Tree for Light Source Sampling
        // 	auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(m_Scene.triangles, m_Scene.materials, m_Scene.worldVertices);
        // 	if(lightTreeEmitterNodes.empty())
        // 		meshPtr->lightTree_blas.nodeCount = 0;
        // 	else
        // 		meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(), static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        // }

        //	Place Banana
        {
            std::vector<Vertex> meshVertices;
            std::vector<uint32_t> meshIndices;
            std::string filePath = "Assets/3D Models/Test/banana.obj";
            Mesh::GenerateMesh(filePath, meshVertices, meshIndices, false);

            //	Set transforms
            glm::vec3 pos{0, 0, 0};
            glm::vec3 rot{-90, 0, 0};
            glm::vec3 scale{1, 1, 1};

            //	Init mesh into scene
            Mesh* meshPtr = m_Scene.AddNewMeshToScene(meshVertices,
                                                      meshIndices,
                                                      pos,
                                                      rot,
                                                      scale,
                                                      7);

            //  Get actual file name. Example : "banana.obj"
            std::filesystem::path p(filePath);
            meshPtr->meshName = p.filename().string();
            
            //	Build BVH for ray collision
            uint32_t triOffset = 0;
            auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, &triOffset);
            meshPtr->blas.objectOffset = triOffset;
            meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

            //	Build Light Tree for Light Source Sampling
            auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(
                m_Scene.triangles, m_Scene.materials, m_Scene.worldVertices);
            if (lightTreeEmitterNodes.empty())
                meshPtr->lightTree_blas.nodeCount = 0;
            else
                meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                           static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        }
        std::vector<Vertex> boxVertices = {
            // Bottom (-Y)
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0, 0}},
            {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1, 0}},
            {{0.5f, -0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {1, 1}},
            {{-0.5f, -0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}, {0, 1}},

            // Top (+Y)
            {{-0.5f, 0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0, 0}},
            {{0.5f, 0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1, 0}},
            {{0.5f, 0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}, {1, 1}},
            {{-0.5f, 0.5f, 0.5f}, {0.0f, -1.0f, 0.0f}, {0, 1}},

            // Front (+Z)
            {{-0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, -1.0f}, {0, 0}},
            {{0.5f, -0.5f, 0.5f}, {0.0f, 0.0f, -1.0f}, {1, 0}},
            {{0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, -1.0f}, {1, 1}},
            {{-0.5f, 0.5f, 0.5f}, {0.0f, 0.0f, -1.0f}, {0, 1}},

            // Back (-Z)
            {{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0, 0}},
            {{0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1, 0}},
            {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {1, 1}},
            {{-0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0, 1}},

            // Left (-X)
            {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0, 0}},
            {{-0.5f, -0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1, 0}},
            {{-0.5f, 0.5f, 0.5f}, {1.0f, 0.0f, 0.0f}, {1, 1}},
            {{-0.5f, 0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0, 1}},

            // Right (+X)
            {{0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0, 0}},
            {{0.5f, -0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}, {1, 0}},
            {{0.5f, 0.5f, 0.5f}, {-1.0f, 0.0f, 0.0f}, {1, 1}},
            {{0.5f, 0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {0, 1}},
        };
        //	Place box
        {
            std::vector<uint32_t> boxIndices = {
                // Bottom
                0, 1, 2, 0, 2, 3,
            };

            //	Set transforms
            glm::vec3 pos{0, -1, 0};
            glm::vec3 rot{0, 0, 0};
            glm::vec3 scale{20, 20, 20};

            //	Init mesh into scene
            Mesh* meshPtr = m_Scene.AddNewMeshToScene(boxVertices,
                                                      boxIndices,
                                                      pos,
                                                      rot,
                                                      scale,
                                                      5);

            meshPtr->meshName = "bottomBox";

            //	Build BVH for ray collision
            uint32_t triOffset = 0;
            auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, &triOffset);
            meshPtr->blas.objectOffset = triOffset;
            meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

            //	Build Light Tree for Light Source Sampling
            auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(
                m_Scene.triangles, m_Scene.materials, m_Scene.worldVertices);
            if (lightTreeEmitterNodes.empty())
                meshPtr->lightTree_blas.nodeCount = 0;
            else
                meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                           static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        }
        {
            std::vector<uint32_t> boxIndices = {
                // Top
                4, 6, 5, 4, 7, 6,
            };

            //	Set transforms
            glm::vec3 pos{0, -1, 0};
            glm::vec3 rot{0, 0, 0};
            glm::vec3 scale{20, 20, 20};

            //	Init mesh into scene
            Mesh* meshPtr = m_Scene.AddNewMeshToScene(boxVertices,
                                                      boxIndices,
                                                      pos,
                                                      rot,
                                                      scale,
                                                      5);

            meshPtr->meshName = "topBox";

            //	Build BVH for ray collision
            uint32_t triOffset = 0;
            auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, &triOffset);
            meshPtr->blas.objectOffset = triOffset;
            meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

            //	Build Light Tree for Light Source Sampling
            auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(
                m_Scene.triangles, m_Scene.materials, m_Scene.worldVertices);
            if (lightTreeEmitterNodes.empty())
                meshPtr->lightTree_blas.nodeCount = 0;
            else
                meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                           static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        }
        {
            std::vector<uint32_t> boxIndices = {
                // Front
                8, 9, 10, 8, 10, 11,
            };

            //	Set transforms
            glm::vec3 pos{0, -1, 0};
            glm::vec3 rot{0, 0, 0};
            glm::vec3 scale{20, 20, 20};

            //	Init mesh into scene
            Mesh* meshPtr = m_Scene.AddNewMeshToScene(boxVertices,
                                                      boxIndices,
                                                      pos,
                                                      rot,
                                                      scale,
                                                      3);

            meshPtr->meshName = "frontBox";

            //	Build BVH for ray collision
            uint32_t triOffset = 0;
            auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, &triOffset);
            meshPtr->blas.objectOffset = triOffset;
            meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

            //	Build Light Tree for Light Source Sampling
            auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(
                m_Scene.triangles, m_Scene.materials, m_Scene.worldVertices);
            if (lightTreeEmitterNodes.empty())
                meshPtr->lightTree_blas.nodeCount = 0;
            else
                meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                           static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        }
        {
            std::vector<uint32_t> boxIndices = {
                // Back
                12, 14, 13, 12, 15, 14,
            };

            //	Set transforms
            glm::vec3 pos{0, -1, 0};
            glm::vec3 rot{0, 0, 0};
            glm::vec3 scale{20, 20, 20};

            //	Init mesh into scene
            Mesh* meshPtr = m_Scene.AddNewMeshToScene(boxVertices,
                                                      boxIndices,
                                                      pos,
                                                      rot,
                                                      scale,
                                                      0);

            meshPtr->meshName = "backBox";

            //	Build BVH for ray collision
            uint32_t triOffset = 0;
            auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, &triOffset);
            meshPtr->blas.objectOffset = triOffset;
            meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

            //	Build Light Tree for Light Source Sampling
            auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(
                m_Scene.triangles, m_Scene.materials, m_Scene.worldVertices);
            if (lightTreeEmitterNodes.empty())
                meshPtr->lightTree_blas.nodeCount = 0;
            else
                meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                           static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        }
        {
            std::vector<uint32_t> boxIndices = {
                // Left
                16, 17, 18, 16, 18, 19,
            };

            //	Set transforms
            glm::vec3 pos{0, -1, 0};
            glm::vec3 rot{0, 0, 0};
            glm::vec3 scale{20, 20, 20};

            //	Init mesh into scene
            Mesh* meshPtr = m_Scene.AddNewMeshToScene(boxVertices,
                                                      boxIndices,
                                                      pos,
                                                      rot,
                                                      scale,
                                                      6);

            meshPtr->meshName = "leftBox";

            //	Build BVH for ray collision
            uint32_t triOffset = 0;
            auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, &triOffset);
            meshPtr->blas.objectOffset = triOffset;
            meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

            //	Build Light Tree for Light Source Sampling
            auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(
                m_Scene.triangles, m_Scene.materials, m_Scene.worldVertices);
            if (lightTreeEmitterNodes.empty())
                meshPtr->lightTree_blas.nodeCount = 0;
            else
                meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                           static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        }
        {
            std::vector<uint32_t> boxIndices = {
                // Right
                20, 22, 21, 20, 23, 22
            };

            //	Set transforms
            glm::vec3 pos{0, -1, 0};
            glm::vec3 rot{0, 0, 0};
            glm::vec3 scale{20, 20, 20};

            //	Init mesh into scene
            Mesh* meshPtr = m_Scene.AddNewMeshToScene(boxVertices,
                                                      boxIndices,
                                                      pos,
                                                      rot,
                                                      scale,
                                                      4);

            meshPtr->meshName = "rightBox";

            //	Build BVH for ray collision
            uint32_t triOffset = 0;
            auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, &triOffset);
            meshPtr->blas.objectOffset = triOffset;
            meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

            //	Build Light Tree for Light Source Sampling
            auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(
                m_Scene.triangles, m_Scene.materials, m_Scene.worldVertices);
            if (lightTreeEmitterNodes.empty())
                meshPtr->lightTree_blas.nodeCount = 0;
            else
                meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                           static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        }
        for (int i = -4; i < 4; i++)
        {
            std::vector<Vertex> planeVertices = {
                {{-0.5f, 0.0f, -0.5f}, {0, 1, 0}, {0, 0}}, // 0: Bottom Left
                {{0.5f, 0.0f, -0.5f}, {0, 1, 0}, {1, 0}}, // 1: Bottom Right
                {{0.5f, 0.0f, 0.5f}, {0, 1, 0}, {1, 1}}, // 2: Top Right
                {{-0.5f, 0.0f, 0.5f}, {0, 1, 0}, {0, 1}}, // 3: Top Left
            };
            std::vector<uint32_t> planeIndices = {
                0, 1, 2, // First triangle
                0, 2, 3 // Second triangle
            };

            //	Set transforms
            glm::vec3 pos{i * 2, 8.99999f, 0};
            glm::vec3 rot{180, 0, 0};
            glm::vec3 scale{1.5, 1.5, 1.5};

            //	Init mesh into scene
            Mesh* meshPtr = m_Scene.AddNewMeshToScene(planeVertices,
                                                      planeIndices,
                                                      pos,
                                                      rot,
                                                      scale,
                                                      2);

            meshPtr->meshName = "lightPlane";

            //	Build BVH for ray collision
            uint32_t triOffset = 0;
            auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, &triOffset);
            meshPtr->blas.objectOffset = triOffset;
            meshPtr->blas.ConstructBVH_SAH(blasObjectNodes.data(), blasObjectNodes.size());

            //	Build Light Tree for Light Source Sampling
            auto lightTreeEmitterNodes = meshPtr->CreateLightTreenodesFromEmmisiveMeshTriangles(
                m_Scene.triangles, m_Scene.materials, m_Scene.worldVertices);
            if (lightTreeEmitterNodes.empty())
                meshPtr->lightTree_blas.nodeCount = 0;
            else
                meshPtr->lightTree_blas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                           static_cast<uint32_t>(lightTreeEmitterNodes.size()));
        }
        
        //	Scene TLAS Construction
        auto tlasObjectNodes = m_Scene.CreateBVHnodesFromSceneMeshes();
        m_Scene.tlas.ConstructBVH_SAH(tlasObjectNodes.data(), tlasObjectNodes.size());

        //	Init Scene Emissive Light Source list
        m_Scene.InitSceneEmissiveTriangles();

        //	Scene Light Tree TLAS Construction
        auto lightTreeEmitterNodes = m_Scene.CreateLightTreeNodesFromBLASLightTrees();
        //auto lightTreeEmitterNodes = m_Scene.CreateLightTreeNodesFromEmissiveTriangles();
        m_Scene.lightTree_tlas.ConstructLightTree(lightTreeEmitterNodes.data(),
                                                  static_cast<uint32_t>(lightTreeEmitterNodes.size()));

        //	Set Starting Camera Position and Direction
        m_Camera.SetPosition(glm::vec3{-0.206f, 8.288f, -9.494f});
        m_Camera.SetDirection(glm::vec3{0.275f, -0.622f, 0.733f});
    }

    void OnUpdate(float ts) override
    {
        bool cameraMoved = m_Camera.OnUpdate(ts);
        if (cameraMoved)
        {
            m_Renderer.ResetFrameIndex();
            m_RenderTime = 0.0f;
            stopRender = false;
        }
    }

    void OnUIRender() override
    {
        ImGui::Begin("General");
        bool isCameraUpdated = false;
        isCameraUpdated |= ImGui::DragFloat3("Camera Position", glm::value_ptr(m_Camera.GetPosition()), 0.1f);
        isCameraUpdated |= ImGui::DragFloat3("Camera Forward Direction", glm::value_ptr(m_Camera.GetDirection()), 0.1f);
        if (isCameraUpdated)
        {
            m_Camera.UpdateCameraView();
            m_Renderer.ResetFrameIndex();
            m_RenderTime = 0.0f;
            stopRender = false;
        }
        bool rayTracingSettingsUpdated = false;
        rayTracingSettingsUpdated |= ImGui::ColorEdit3("Skybox Color", glm::value_ptr(m_Renderer.GetSettings().skyColor));
        rayTracingSettingsUpdated |= ImGui::DragInt("Light Bounce Amount", &m_Renderer.GetSettings().lightBounces, 1.0f,
                                    0, UINT8_MAX);
        rayTracingSettingsUpdated |= ImGui::DragInt("Ray Sample Count", &m_Renderer.GetSettings().sampleCount, 1.0f, 1,
                                                    UINT8_MAX);
        ImGui::Text("Resolution : %dx%d", m_ViewportWidth, m_ViewportHeight);
        ImGui::Text("Triangle Count : %d", static_cast<uint32_t>(m_Scene.triangles.size()));
        ImGui::Text("Vertices Count : %d", static_cast<uint32_t>(m_Scene.vertices.size()));
        ImGui::Text("Frame Time : %.3fms", m_CurrentFrameTime);
        ImGui::DragFloat("Total Time to Render(min)", &m_TimeToRender, 0.1f);
        ImGui::Text("Render Time : %.3f min(s)", m_RenderTime / 60000.0f);
        ImGui::Text("Accumulated Frames : %d", m_Renderer.GetCurrentFrameIndex());

        //	Sampling Technique Selection
        int currentIndex = m_Renderer.GetSettings().currentSamplingTechnique;

        if (ImGui::Combo("Sampling Technique", &currentIndex, samplingTechniqueNames.data(),
                         static_cast<int>(samplingTechniqueNames.size())))
        {
            if (m_Renderer.GetSettings().currentSamplingTechnique != static_cast<SamplingTechniqueEnum>(currentIndex))
            {
                m_Renderer.ResetFrameIndex();
                m_RenderTime = 0.0f;
                stopRender = false;
            }

            m_Renderer.GetSettings().currentSamplingTechnique = static_cast<SamplingTechniqueEnum>(currentIndex);
        }
        
        if (ImGui::Button("Reset"))
        {
            m_Renderer.ResetFrameIndex();
            m_RenderTime = 0.0f;
            stopRender = false;
        }

        if (ImGui::Button("Save render image"))
        {
            SaveRenderImage();
        }
        
        if (ImGui::Button("Benchmark render results"))
        {
            const char* patterns[] = { "*.bmp" };
            const char* abs = tinyfd_openFileDialog(
                "Select Reference Image",
                "",
                1, patterns,
                NULL, 0);

            if (abs)
            {
                auto rel = std::filesystem::relative(abs, std::filesystem::current_path()).string();
                Texture referenceImage = Texture(rel);

                if(m_ViewportHeight == referenceImage.height && m_ViewportWidth == referenceImage.width)
                {
                    SaveBenchmarkResults(referenceImage.pixels);
                }
                else
                {
                    std::cerr << "Incorrect Reference Image Dimensions!\n";
                }
                referenceImage.FreeTexture();
            }
        }
        ImGui::SameLine();
        ImGui::Text("Note: You will be prompt to select a reference image to compare with current render");

        ImGui::Checkbox("Accumulate (Not good for Dynamic Scenes! Only use when trying to get a good picture)",
                        &m_Renderer.GetSettings().toAccumulate);
        ImGui::End();

        ImGui::Begin("ReSTIR");
        rayTracingSettingsUpdated |= ImGui::DragInt("ReSTIR Candidate Count",
                                                    &m_Renderer.GetSettings().lightCandidateCount, 1.0f, 1, UINT16_MAX);
        rayTracingSettingsUpdated |= ImGui::DragInt("ReSTIR Temporal History Limit",
                                            &m_Renderer.GetSettings().temporalHistoryLimit, 1.0f, 1, UINT8_MAX);
        rayTracingSettingsUpdated |= ImGui::DragInt("ReSTIR Spatial Neighbour Count",
                                            &m_Renderer.GetSettings().spatialNeighborNum, 1.0f, 1, UINT8_MAX);
        rayTracingSettingsUpdated |= ImGui::DragInt("ReSTIR Spatial Neighbour Radius",
                                    &m_Renderer.GetSettings().spatialNeighborRadius, 1.0f, 1, UINT8_MAX);
        rayTracingSettingsUpdated |= ImGui::Checkbox("ReSTIR Temporal Reuse",
                        &m_Renderer.GetSettings().useTemporalReuse);
        rayTracingSettingsUpdated |= ImGui::Checkbox("ReSTIR Spatial Reuse",
                        &m_Renderer.GetSettings().useSpatialReuse);
        ImGui::End();

        if (m_Renderer.GetSettings().toAccumulate && rayTracingSettingsUpdated)
        {
            m_Renderer.ResetFrameIndex();
            m_RenderTime = 0.0f;
            stopRender = false;
        }
        
        ImGui::Begin("Scene");
        for (uint32_t i = 0; i < m_Scene.meshes.size(); i++)
        {
            ImGui::PushID(i);

            Mesh& mesh = m_Scene.meshes[i];
            bool meshTransformToBeUpdated = false, meshMatToBeUpdated = false;
            ImGui::Text("Object ID : %d", i);
            ImGui::Text("Object Name : %s", mesh.meshName);
            meshTransformToBeUpdated |= ImGui::DragFloat3("Position", glm::value_ptr(mesh.position), 0.1f);
            meshTransformToBeUpdated |= ImGui::DragFloat3("Rotation", glm::value_ptr(mesh.rotation), 0.1f);
            meshTransformToBeUpdated |= ImGui::DragFloat3("Scale", glm::value_ptr(mesh.scale), 0.1f);
            meshMatToBeUpdated |= ImGui::DragInt("Material Index", &mesh.materialIndex, 1.0f, 0,
                                                 (int)m_Scene.materials.size() - 1);

            if (meshMatToBeUpdated || meshTransformToBeUpdated)
            {
                m_Scene.sceneManager.meshesToUpdate.emplace_back(meshTransformToBeUpdated, meshMatToBeUpdated, i);
                //std::cerr << "object index " << i << " is updated\n";
            }
            ImGui::Separator();
            ImGui::PopID();
        }
        if (ImGui::Button("Import Mesh"))
        {
            // File filters
            const char* meshPatterns[] = { "*.obj", "*.fbx", "*.gltf", "*.glb" };

            // Open native file dialog
            const char* absPath = tinyfd_openFileDialog(
                "Select Mesh",
                "",
                4, meshPatterns,
                nullptr,
                0 
            );

            if (absPath) // user selected a file
            {
                // Convert absolute to relative path
                std::filesystem::path absolute(absPath);
                std::filesystem::path base = std::filesystem::current_path();
                std::string relativePath = std::filesystem::relative(absolute, base).string();

                // Send into your engine
                m_Scene.CreateNewMeshInScene(relativePath);
                m_Renderer.SetSceneToBeUpdatedFlag(true);
            }
        }
        ImGui::End();

        ImGui::Begin("Materials");
        for (uint32_t i = 0; i < m_Scene.materials.size(); i++)
        {
            ImGui::PushID(i);

            Material& material = m_Scene.materials[i];
            bool matToBeUpdated = false;
            ImGui::Text("Material ID : %d", i);
            matToBeUpdated |= ImGui::ColorEdit3("Albedo", glm::value_ptr(material.albedo));
            uint32_t minIndex = 0;
            uint32_t maxIndex = static_cast<uint32_t>(m_Scene.textures.size()) - 1;
            matToBeUpdated |= ImGui::DragScalar("Albedo Map Index", ImGuiDataType_U32, &material.albedoMapIndex, 1.0f, &minIndex, &maxIndex);
            matToBeUpdated |= ImGui::Checkbox("Use Albedo Map instead of Solid Color", &material.isUseAlbedoMap);
            matToBeUpdated |= ImGui::DragFloat("Roughness", &material.roughness, 0.05f, 0.0f, 1.0f);
            matToBeUpdated |= ImGui::DragFloat("Metallic", &material.metallic, 0.05f, 0.0f, 1.0f);
            matToBeUpdated |= ImGui::ColorEdit3("Emission Color", glm::value_ptr(material.emissionColor));
            matToBeUpdated |= ImGui::DragFloat("Emission Power", &material.emissionPower, 100.f, 0.0f, FLT_MAX);

            if (matToBeUpdated)
                m_Scene.sceneManager.materialsToUpdate.push_back(i);
            ImGui::Separator();
            ImGui::PopID();
        }
        if (ImGui::Button("Create New Material"))
        {
            m_Scene.CreateNewMaterialInScene();
            m_Renderer.SetSceneToBeUpdatedFlag(true);
        }
        ImGui::End();

        ImGui::Begin("Textures");
        for (uint32_t i = 0; i < m_Scene.textures.size(); i++)
        {
            ImGui::PushID(i);

            ImGui::Text("Texture ID : %d", i);
            ImGui::Text("File Name : %s", m_Scene.textures[i].fileName.c_str());
            ImGui::Text("Width : %d", m_Scene.textures[i].width);
            ImGui::Text("Height : %d", m_Scene.textures[i].height);
            
            ImGui::Separator();
            ImGui::PopID();
        }
        if (ImGui::Button("Import Texture"))
        {
            const char* patterns[5] = { "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tga" };
            const char* abs = tinyfd_openFileDialog(
                "Select Texture",
                "",
                5, patterns,
                NULL, 0);

            if (abs)
            {
                auto rel = std::filesystem::relative(abs, std::filesystem::current_path()).string();
                m_Scene.CreateNewTextureInScene(rel);
                m_Renderer.SetSceneToBeUpdatedFlag(true);
            }
        }
        
        ImGui::End();

        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("Viewport");

        m_ViewportWidth = ImGui::GetContentRegionAvail().x;
        m_ViewportHeight = ImGui::GetContentRegionAvail().y;

        //	display image on window
        auto image = m_Renderer.GetFinalRenderImage();
        if (image)
        {
            //	uv0 and uv1 paramters can be used to flip the UV coords
            ImGui::Image(image->GetDescriptorSet(), {(float)image->GetWidth(), (float)image->GetHeight()},
                         ImVec2(0, 1), ImVec2(1, 0));
        }

        ImGui::End();
        ImGui::PopStyleVar();
        
        m_Scene.sceneManager.PerformAllSceneUpdates(m_Scene, m_Renderer);
        Render();
    }

    void SaveRenderImage()
    {
        if (m_Renderer.GetCurrentFrameIndex() == 1)
            m_AverageFrameTime = m_CurrentFrameTime;
        else
            m_AverageFrameTime = m_RenderTime / (float)m_Renderer.GetCurrentFrameIndex();

        std::string fileName = "RenderedImages/output";
        fileName.append("_" + std::to_string(m_AverageFrameTime) + "(ms)");
        fileName.append("_" + std::to_string(m_RenderTime / 60000.0f) + "(min)s");
        std::string samplingTechniqueName = samplingTechniqueNames[m_Renderer.GetSettings().currentSamplingTechnique];
        fileName.append("_" + samplingTechniqueName);
        
        if(m_Renderer.GetSettings().currentSamplingTechnique != RESTIR_DI && m_Renderer.GetSettings().currentSamplingTechnique != RESTIR_GI)
        {
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().sampleCount) + "sample(s)");
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().lightBounces) + "rayBounces(s)");
        }
        else if(m_Renderer.GetSettings().currentSamplingTechnique == RESTIR_DI)
        {
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().lightCandidateCount) + "candidate(s)");
            if(m_Renderer.GetSettings().useTemporalReuse)
                fileName.append("_temporalHistoryLimit(" + std::to_string(m_Renderer.GetSettings().temporalHistoryLimit) + ")");
            if(m_Renderer.GetSettings().useSpatialReuse)
            {
                fileName.append("_NeighbourCount(" + std::to_string(m_Renderer.GetSettings().spatialNeighborNum) + ")");
                fileName.append("_NeighbourRadius(" + std::to_string(m_Renderer.GetSettings().spatialNeighborRadius) + ")");
            }
        }
        else if(m_Renderer.GetSettings().currentSamplingTechnique == RESTIR_GI)
        {
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().lightCandidateCount) + "candidate(s)");
            if(m_Renderer.GetSettings().useTemporalReuse)
                fileName.append("_temporalHistoryLimit(" + std::to_string(m_Renderer.GetSettings().temporalHistoryLimit) + ")");
            if(m_Renderer.GetSettings().useSpatialReuse)
            {
                fileName.append("_NeighbourCount(" + std::to_string(m_Renderer.GetSettings().spatialNeighborNum) + ")");
                fileName.append("_NeighbourRadius(" + std::to_string(m_Renderer.GetSettings().spatialNeighborRadius) + ")");
            }
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().lightBounces) + "rayBounces(s)");
        }
        
        std::string finalFilename = MisUtils::GetTimestampedFilename(fileName);
        MisUtils::SaveABGRToBMP(finalFilename, m_Renderer.GetRenderImageDataPtr(), m_ViewportWidth, m_ViewportHeight);
    }

    void SaveBenchmarkResults(uint32_t* referenceImagePixels)
    {
        if (m_Renderer.GetCurrentFrameIndex() == 1)
            m_AverageFrameTime = m_CurrentFrameTime;
        else
            m_AverageFrameTime = m_RenderTime / (float)m_Renderer.GetCurrentFrameIndex();

        std::string fileName = "RenderedImages/output";
        fileName.append("_" + std::to_string(m_AverageFrameTime) + "(ms)");
        fileName.append("_" + std::to_string(m_RenderTime / 60000.0f) + "(min)s");
        std::string samplingTechniqueName = samplingTechniqueNames[m_Renderer.GetSettings().currentSamplingTechnique];
        fileName.append("_" + samplingTechniqueName);
        
        if(m_Renderer.GetSettings().currentSamplingTechnique != RESTIR_DI && m_Renderer.GetSettings().currentSamplingTechnique != RESTIR_GI)
        {
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().sampleCount) + "sample(s)");
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().lightBounces) + "rayBounces(s)");
        }
        else if(m_Renderer.GetSettings().currentSamplingTechnique == RESTIR_DI)
        {
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().lightCandidateCount) + "candidate(s)");
            if(m_Renderer.GetSettings().useTemporalReuse)
                fileName.append("_temporalHistoryLimit(" + std::to_string(m_Renderer.GetSettings().temporalHistoryLimit) + ")");
            if(m_Renderer.GetSettings().useSpatialReuse)
            {
                fileName.append("_NeighbourCount(" + std::to_string(m_Renderer.GetSettings().spatialNeighborNum) + ")");
                fileName.append("_NeighbourRadius(" + std::to_string(m_Renderer.GetSettings().spatialNeighborRadius) + ")");
            }
        }
        else if(m_Renderer.GetSettings().currentSamplingTechnique == RESTIR_GI)
        {
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().lightCandidateCount) + "candidate(s)");
            if(m_Renderer.GetSettings().useTemporalReuse)
                fileName.append("_temporalHistoryLimit(" + std::to_string(m_Renderer.GetSettings().temporalHistoryLimit) + ")");
            if(m_Renderer.GetSettings().useSpatialReuse)
            {
                fileName.append("_NeighbourCount(" + std::to_string(m_Renderer.GetSettings().spatialNeighborNum) + ")");
                fileName.append("_NeighbourRadius(" + std::to_string(m_Renderer.GetSettings().spatialNeighborRadius) + ")");
            }
            fileName.append("_" + std::to_string(m_Renderer.GetSettings().lightBounces) + "rayBounces(s)");
        }
        
        //  Add noise metrics in the log by first loading a reference image from file explorer and compare it to get MSE and PSNR
        float mse = MisUtils::ComputeMSE(referenceImagePixels, m_Renderer.GetRenderImageDataPtr(), m_ViewportWidth, m_ViewportHeight);
        fileName.append("_MSE(" + std::to_string(mse) + ")");
        float psnr = MisUtils::ComputePSNR(mse);
        fileName.append("_PSNR(" + std::to_string(psnr) + ")");
        
        std::string finalFilename = MisUtils::GetTimestampedFilename(fileName);
        MisUtils::SaveABGRToBMP(finalFilename, m_Renderer.GetRenderImageDataPtr(), m_ViewportWidth, m_ViewportHeight);
    }

    void Render()
    {
        Walnut::Timer timer;

        m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);
        m_Camera.OnResize(m_ViewportWidth, m_ViewportHeight);
        if (!stopRender)
        {
            m_Renderer.Render(m_Scene, m_Camera);

            if (m_Renderer.GetSettings().toAccumulate)
            {
                m_CurrentFrameTime = timer.ElapsedMillis();
                m_RenderTime += m_CurrentFrameTime;
            }
            else
            {
                m_CurrentFrameTime = timer.ElapsedMillis();
                m_RenderTime = 0.0f;
            }
        }

        //	OFFLINE RENDERING
        if (m_RenderTime / 60000.0f >= m_TimeToRender && !stopRender)
        {
            stopRender = true;
            SaveRenderImage();
        }

        //  Set Prev buffers
        m_Camera.SetPrevProjection(m_Camera.GetProjection());
        m_Camera.SetPrevView(m_Camera.GetView());
    }
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
    Walnut::ApplicationSpecification spec;
    spec.Name = "FYP Ray Tracer";

    auto app = new Walnut::Application(spec);
    app->PushLayer<MainLayer>();
    app->SetMenubarCallback([app]()
    {
        if (ImGui::BeginMenu("File"))
        {
            if (ImGui::MenuItem("Exit"))
            {
                app->Close();
            }
            ImGui::EndMenu();
        }
    });
    return app;
}
