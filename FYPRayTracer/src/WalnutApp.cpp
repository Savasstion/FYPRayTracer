#include "Classes/BaseClasses/Camera.h"
#include "Walnut/Application.h"
#include "Walnut/EntryPoint.h"
#include "Walnut/Image.h"
#include "Walnut/Timer.h"
#include "Classes/Core/Renderer.h"
#include <glm/gtc/type_ptr.hpp>

class ExampleLayer : public Walnut::Layer
{
private:
	uint32_t m_ViewportWidth = 0, m_ViewportHeight = 0;
	Renderer m_Renderer;
	float m_RenderTime = 0.0f;
	Camera m_Camera;
	Scene m_Scene;
	
public:
	ExampleLayer()
	: m_Camera(45.0f, 0.1f, 100.0f)
	{
		Material& matPinkSphere = m_Scene.materials.emplace_back();
		matPinkSphere.albedo = {1.0f,0.0f,1.0f};
		matPinkSphere.roughness = 0.0f;
		
		Material& matBlueSphere = m_Scene.materials.emplace_back();
		matBlueSphere.albedo = {0.2f,0.3f,1.0f};
		matBlueSphere.roughness = 0.1f;
		
		{
			Sphere sphere;
			sphere.position = {0,0,0};
			sphere.radius = 1.0f;
			sphere.materialIndex = 0;

			m_Scene.spheres.push_back(sphere);
		}
		
		{
			Sphere sphere;
			sphere.position = {0,-101,0};
			sphere.radius = 100.0f;
			sphere.materialIndex = 1;

			m_Scene.spheres.push_back(sphere);
		}

	}
	
	virtual void OnUpdate(float ts) override
	{
		bool cameraMoved = m_Camera.OnUpdate(ts);
		if(cameraMoved)
			m_Renderer.ResetFrameIndex();
	}
	
	virtual void OnUIRender() override
	{
		ImGui::Begin("Settings");
		ImGui::Text("Render Time : %.3fms", m_RenderTime);
		//if (ImGui::Button("Render"))
		//		Render();
		if (ImGui::Button("Reset"))
			m_Renderer.ResetFrameIndex();
		ImGui::Checkbox("Accumulate (Not good for Dynamic Scenes! Only use when trying to get a good picture)", &m_Renderer.GetSettings().toAccumulate);
		ImGui::End();

		ImGui::Begin("Scene");
		for(size_t i = 0; i < m_Scene.spheres.size(); i++)
		{
			ImGui::PushID(i);

			Sphere& sphere = m_Scene.spheres[i];
			ImGui::DragFloat3("Position", glm::value_ptr(sphere.position), 0.1f);
			ImGui::DragFloat("Radius", &sphere.radius, 0.1f);
			ImGui::DragInt("Material Index", &sphere.materialIndex, 1.0f, 0, (int)m_Scene.materials.size()-1);
			
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
		m_Renderer.Render(m_Scene, m_Camera);
		
		m_RenderTime = timer.ElapsedMillis();
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