#include "Classes/BaseClasses/Camera.h"
#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Walnut/Image.h"
#include "Walnut/Timer.h"
#include "Classes/Core/Renderer.h"
#include <glm/gtc/type_ptr.hpp>
#include "Utility/MisUtils.h"

class ExampleLayer : public Walnut::Layer
{
private:
	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
	Renderer m_Renderer;
	float m_CurrentFrameTime = 0.0f;
	float m_AverageFrameTime = 0.0f;
	float m_TimeToRender = 15.0f;
	float m_RenderTime = 0.0f;
	Camera m_Camera;
	Scene m_Scene;

	bool stopDemo = false;
	
public:
	ExampleLayer()
	: m_Camera(45.0f, 0.1f, 100.0f)
	{
		Material& matPinkSphere = m_Scene.materials.emplace_back();
		matPinkSphere.albedo = {1.0f,0.0f,1.0f};
		matPinkSphere.roughness = 0.1f;
		matPinkSphere.metallic = 0.45f;
		
		Material& matBlueSphere = m_Scene.materials.emplace_back();
		matBlueSphere.albedo = {0.2f,0.3f,1.0f};
		matBlueSphere.roughness = 0.7f;
		matBlueSphere.metallic = 0.0f;
		

		Material& matWhiteGlowingSphere = m_Scene.materials.emplace_back();
		matWhiteGlowingSphere.albedo = {1,1,1};
		matWhiteGlowingSphere.roughness = 0.1f;
		matWhiteGlowingSphere.emissionColor = matWhiteGlowingSphere.albedo;
		matWhiteGlowingSphere.emissionPower = 20.0f;
		
		{
			std::vector<Vertex> planeVertices = {
				{{-0.5f, 0.0f, -0.5f}, {0, 1, 0}, {0, 0}}, // 0: Bottom Left
				{{ 0.5f, 0.0f, -0.5f}, {0, 1, 0}, {1, 0}}, // 1: Bottom Right
				{{ 0.5f, 0.0f,  0.5f}, {0, 1, 0}, {1, 1}}, // 2: Top Right
				{{-0.5f, 0.0f,  0.5f}, {0, 1, 0}, {0, 1}}, // 3: Top Left
			};
			std::vector<uint32_t> planeIndices = {
				0, 1, 2,  // First triangle
				0, 2, 3   // Second triangle
			};
		
			glm::vec3 pos{ 0,-1,0 };
			glm::vec3 rot{ 0,0,0 };
			glm::vec3 scale{ 10,10,10 };
			Mesh* meshPtr = m_Scene.AddNewMeshToScene(planeVertices,
				planeIndices,
				pos,
				rot,
				scale,
				0);
		
			auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, meshPtr->indexStart / 3, meshPtr->indexCount / 3);
			meshPtr->blas.CUDA_ConstructBVHInParallel(blasObjectNodes.data(), blasObjectNodes.size());
		}
		// {
		// 	std::vector<Vertex> sphereVertices;
		// 	std::vector<uint32_t> sphereIndices;
		//
		// 	Mesh::GenerateSphereMesh(1, 20,20, sphereVertices, sphereIndices);
		// 	glm::vec3 pos{0,1,0};
		// 	glm::vec3 rot{0,0,0};
		// 	glm::vec3 scale{1,1,1};
		// 	Mesh* meshPtr = m_Scene.AddNewMeshToScene(sphereVertices,
		// 					sphereIndices,
		// 					pos,
		// 					rot,
		// 					scale,
		// 					1);
		//
		// 	auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, meshPtr->indexStart / 3, meshPtr->indexCount / 3);
		// 	meshPtr->blas.CUDA_ConstructBVHInParallel(blasObjectNodes.data(), blasObjectNodes.size());
		// }
		{
			std::vector<Vertex> planeVertices = {
				{{-0.5f, 0.0f, -0.5f}, {0, 1, 0}, {0, 0}}, // 0: Bottom Left
				{{ 0.5f, 0.0f, -0.5f}, {0, 1, 0}, {1, 0}}, // 1: Bottom Right
				{{ 0.5f, 0.0f,  0.5f}, {0, 1, 0}, {1, 1}}, // 2: Top Right
				{{-0.5f, 0.0f,  0.5f}, {0, 1, 0}, {0, 1}}, // 3: Top Left
			};
			std::vector<uint32_t> planeIndices = {
				0, 1, 2,  // First triangle
				0, 2, 3   // Second triangle
			};
		
			glm::vec3 pos{ 0,5,0 };
			glm::vec3 rot{ 180,0,0 };
			glm::vec3 scale{ 10,10,10 };
			Mesh* meshPtr = m_Scene.AddNewMeshToScene(planeVertices,
				planeIndices,
				pos,
				rot,
				scale,
				1);
		
			auto blasObjectNodes = meshPtr->CreateBVHnodesFromMeshTriangles(m_Scene.triangles, meshPtr->indexStart / 3, meshPtr->indexCount / 3);
			meshPtr->blas.CUDA_ConstructBVHInParallel(blasObjectNodes.data(), blasObjectNodes.size());
		}

		
		//	Scene TLAS Construction
		auto tlasObjectNodes = m_Scene.CreateBVHnodesFromSceneMeshes();
		//auto tlasObjectNodes = m_Scene.CreateBVHnodesFromSceneTriangles();
		m_Scene.tlas.CUDA_ConstructBVHInParallel(tlasObjectNodes.data(), tlasObjectNodes.size());
		
	}
	
	virtual void OnUpdate(float ts) override
	{
		bool cameraMoved = m_Camera.OnUpdate(ts);
		if(cameraMoved)
		{
			m_Renderer.ResetFrameIndex();
			m_RenderTime = 0.0f;
		}
		
	}
	
	virtual void OnUIRender() override
	{
		ImGui::Begin("Settings");
		ImGui::ColorEdit3("Emission Color", glm::value_ptr(m_Renderer.GetSettings().skyColor));
		ImGui::DragInt("Light Bounce Amount", &m_Renderer.GetSettings().lightBounces, 1.0f, 0, UINT8_MAX);
		ImGui::DragInt("Ray Sample Count", &m_Renderer.GetSettings().sampleCount, 1.0f, 1, UINT8_MAX);
		ImGui::Text("Resolution : %dx%d", m_ViewportWidth, m_ViewportHeight);
		ImGui::Text("Frame Time : %.3fms", m_CurrentFrameTime);
		ImGui::DragFloat("Total Time to Render(min)", &m_TimeToRender, 0.1f);
		ImGui::Text("Render Time : %.3f min(s)", m_RenderTime / 60000.0f);
		ImGui::Text("Accumulated Frames : %d", m_Renderer.GetCurrentFrameIndex());
		if (ImGui::Button("Reset"))
		{
			m_Renderer.ResetFrameIndex();
			m_RenderTime = 0.0f;
		}

		ImGui::Checkbox("Accumulate (Not good for Dynamic Scenes! Only use when trying to get a good picture)", &m_Renderer.GetSettings().toAccumulate);
		ImGui::End();

		ImGui::Begin("Scene");
		for(size_t i = 0; i < m_Scene.meshes.size(); i++)
		{
			ImGui::PushID(i);

			Mesh& mesh = m_Scene.meshes[i];
			ImGui::DragFloat3("Position", glm::value_ptr(mesh.position), 0.1f);
			ImGui::DragInt("Material Index", &mesh.materialIndex, 1.0f, 0, (int)m_Scene.materials.size()-1);
			
			ImGui::Separator();
			
			ImGui::PopID();
		}

		for(size_t i = 0; i < m_Scene.materials.size(); i++)
		{
			ImGui::PushID(i);

			Material& material = m_Scene.materials[i];
			ImGui::ColorEdit3("Albedo", glm::value_ptr(material.albedo));
			ImGui::DragFloat("Roughness", &material.roughness, 0.05f, 0.0f, 1.0f);
			ImGui::DragFloat("Metallic", &material.metallic, 0.05f, 0.0f, 1.0f);
			ImGui::ColorEdit3("Emission Color", glm::value_ptr(material.emissionColor));
			ImGui::DragFloat("Emission Power", &material.emissionPower, 0.05f, 0.0f, FLT_MAX);

			ImGui::Separator();

			ImGui::PopID();
		}
		
		ImGui::End();

		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f,0.0f));
		ImGui::Begin("Viewport");

		m_ViewportWidth = ImGui::GetContentRegionAvail().x;
		m_ViewportHeight = ImGui::GetContentRegionAvail().y;

		//	display image on window
		auto image = m_Renderer.GetFinalRenderImage();
		if(image)
		{
			//	uv0 and uv1 paramters can be used to flip the UV coords
			ImGui::Image(image->GetDescriptorSet(), {(float)image->GetWidth(), (float)image->GetHeight()},
				ImVec2(0,1), ImVec2(1,0));
		}	

		ImGui::End();
		ImGui::PopStyleVar();

		Render();
	}

	void Render()
	{
		Walnut::Timer timer;
		
		m_Renderer.OnResize(m_ViewportWidth, m_ViewportHeight);
		m_Camera.OnResize(m_ViewportWidth, m_ViewportHeight);
		if(!stopDemo)
		{
			m_Renderer.Render(m_Scene, m_Camera);
			m_CurrentFrameTime = timer.ElapsedMillis();
			m_RenderTime += m_CurrentFrameTime;
		}
		
		//	DEBUG
		 if(m_RenderTime / 60000.0f >= m_TimeToRender && !stopDemo)
		 {
		 	stopDemo = true;
		 	m_AverageFrameTime = m_RenderTime / (float)m_Renderer.GetCurrentFrameIndex();
		 	std::string fileName = "RenderedImages/output";
		 	fileName.append("_" + std::to_string(m_AverageFrameTime) + "(ms)");
		 	fileName.append("_" + std::to_string(m_TimeToRender) + "(min)s");
		 	std::string finalFilename = MisUtils::GetTimestampedFilename(fileName);
		 	MisUtils::SaveABGRToBMP(finalFilename, m_Renderer.GetRenderImageDataPtr(), m_ViewportWidth, m_ViewportHeight);
		 }
	}
};

Walnut::Application* Walnut::CreateApplication(int argc, char** argv)
{
	Walnut::ApplicationSpecification spec;
	spec.Name = "FYP Ray Tracer";

	Walnut::Application* app = new Walnut::Application(spec);
	app->PushLayer<ExampleLayer>();
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