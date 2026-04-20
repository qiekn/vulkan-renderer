module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <imgui.h>

export module app;

import vulkan;
import std;
import engine.event;
import engine.window;
import engine.device;
import engine.swapchain;
import engine.mesh;
import engine.pipeline;
import engine.renderer;
import engine.scene;
import engine.resource;
import engine.render_pass;
import engine.forward_pass;
import engine.uniform;
import engine.camera_controller;
import engine.imgui_layer;
import engine.imgui_pass;

namespace app {

// ---------------------------------------------------------------------------: Constants

constexpr std::uint32_t kWindowWidth = 1280;
constexpr std::uint32_t kWindowHeight = 720;
constexpr std::string_view kWindowTitle = "Vulkan Engine";
constexpr std::string_view kPbrShaderId = "pbr";
constexpr std::string_view kPbrShaderPath = "assets/shaders/slang.spv";

// ---------------------------------------------------------------------------: Listener

// Esc closes, C dumps camera matrices, F1 toggles UI mode (cursor released +
// camera input paused so ImGui owns the pointer), resize updates aspect.
class AppListener : public engine::EventListener {
public:
  AppListener(engine::Window& window, engine::CameraComponent& camera,
              engine::CameraController& controller, bool& ui_mode)
      : window_(window), camera_(camera), controller_(controller), ui_mode_(ui_mode) {}

  void OnEvent(engine::Event& event) override {
    engine::EventDispatcher dispatcher(event);

    dispatcher.Dispatch<engine::KeyPressEvent>([this](engine::KeyPressEvent& e) {
      if (e.GetKeyCode() == GLFW_KEY_ESCAPE) {
        window_.RequestClose();
        return true;
      }
      if (e.GetKeyCode() == GLFW_KEY_F1 && !e.IsRepeat()) {
        ui_mode_ = !ui_mode_;
        window_.SetCursorDisabled(!ui_mode_);
        controller_.SetMouseCaptured(!ui_mode_);
        return true;
      }
      if (e.GetKeyCode() == GLFW_KEY_C && !e.IsRepeat()) {
        auto view = camera_.GetViewMatrix();
        auto proj = camera_.GetProjectionMatrix();
        std::cout << "[camera] view[3]=(" << view[3][0] << ", " << view[3][1] << ", " << view[3][2]
                  << ")  proj[0][0]=" << proj[0][0] << "  proj[1][1]=" << proj[1][1] << '\n';
      }
      return false;
    });

    dispatcher.Dispatch<engine::WindowResizeEvent>([this](engine::WindowResizeEvent& e) {
      if (e.GetWidth() > 0 && e.GetHeight() > 0) {
        camera_.SetAspect(static_cast<float>(e.GetWidth()) / static_cast<float>(e.GetHeight()));
      }
      return false;
    });
  }

private:
  engine::Window& window_;
  engine::CameraComponent& camera_;
  engine::CameraController& controller_;
  bool& ui_mode_;
};

// ---------------------------------------------------------------------------: Application

export class Application {
public:
  void Run() {
    engine::EventBus event_bus;
    engine::Window window(kWindowWidth, kWindowHeight, kWindowTitle, event_bus);
    engine::Device device(window);
    engine::Swapchain swapchain(window, device);

    engine::ResourceManager resources;
    auto pbr_shader = resources.Load<engine::ShaderResource>(
        std::string(kPbrShaderId), device, std::filesystem::path(kPbrShaderPath));
    if (!pbr_shader) {
      throw std::runtime_error("Failed to load pbr shader");
    }

    engine::UniformBufferSet ubo_set(device, engine::Renderer::kMaxFramesInFlight);
    engine::Mesh cube = engine::Mesh::CreateCube(device, 1.0f);

    auto vertex_attributes = engine::Mesh::GetAttributeDescriptions();
    vk::PushConstantRange material_range{
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(engine::MaterialPushConstants),
    };
    engine::PipelineConfig pipeline_config{
        .shader_module = *pbr_shader->GetModule(),
        .color_format = swapchain.GetImageFormat(),
        .depth_format = swapchain.GetDepthFormat(),
        .vertex_binding = engine::Mesh::GetBindingDescription(),
        .vertex_attributes = std::span(vertex_attributes),
        .descriptor_set_layout = *ubo_set.GetLayout(),
        .push_constant_ranges = std::span(&material_range, 1),
        .cull_mode = vk::CullModeFlagBits::eBack,
        .front_face = vk::FrontFace::eCounterClockwise,
    };
    engine::Pipeline pipeline(device, pipeline_config);

    engine::MaterialPushConstants material{
        .base_color = glm::vec4(0.9f, 0.1f, 0.05f, 1.0f),
        .metallic = 0.1f,
        .roughness = 0.35f,
    };

    engine::RenderPassManager pass_manager;
    auto* forward_pass =
        pass_manager.AddPass<engine::ForwardPass>(pipeline, ubo_set, cube, material);
    pass_manager.AddPass<engine::ImGuiPass>();

    // ImGuiLayer builds its graphics pipeline against the swapchain color
    // format, so it has to be alive before Renderer records its first frame.
    engine::ImGuiLayer imgui_layer(window, device, swapchain);

    engine::Renderer renderer(window, device, swapchain, pass_manager);

    engine::Scene scene;

    engine::Entity* cube_entity = scene.CreateEntity("Cube");
    auto* cube_transform = cube_entity->AddComponent<engine::TransformComponent>();

    engine::Entity* camera_entity = scene.CreateEntity("Camera");
    auto* camera_transform = camera_entity->AddComponent<engine::TransformComponent>();
    auto* camera = camera_entity->AddComponent<engine::CameraComponent>();
    camera_transform->SetPosition(glm::vec3(0.0f, 1.2f, 3.5f));
    auto extent = swapchain.GetExtent();
    camera->SetAspect(static_cast<float>(extent.width) / static_cast<float>(extent.height));

    engine::CameraController controller(*camera_transform, *camera);
    event_bus.AddListener(&controller);

    bool ui_mode = false;
    AppListener listener(window, *camera, controller, ui_mode);
    event_bus.AddListener(&listener);

    window.SetCursorDisabled(true);

    // Four point lights in a loose ring around the cube so every face picks up
    // at least one direct light. Radiant intensity is in "watts per steradian"
    // units — attenuation falls off with 1/r^2, so push the values up.
    const std::array<glm::vec4, 4> light_positions = {
        glm::vec4( 2.5f,  2.5f,  2.5f, 1.0f),
        glm::vec4(-2.5f,  2.5f,  2.5f, 1.0f),
        glm::vec4( 2.5f, -1.5f, -2.5f, 1.0f),
        glm::vec4(-2.5f,  1.5f, -2.5f, 1.0f),
    };
    const std::array<glm::vec4, 4> light_colors = {
        glm::vec4(60.0f, 55.0f, 50.0f, 1.0f),  // warm key
        glm::vec4(20.0f, 25.0f, 45.0f, 1.0f),  // cool fill
        glm::vec4(35.0f, 15.0f, 15.0f, 1.0f),  // red rim
        glm::vec4(15.0f, 35.0f, 25.0f, 1.0f),  // green rim
    };

    float exposure = 1.0f;
    float gamma = 2.2f;
    bool show_demo = false;

    auto last_time = std::chrono::steady_clock::now();
    while (!window.ShouldClose()) {
      window.PollEvents();

      auto now = std::chrono::steady_clock::now();
      float delta = std::chrono::duration<float>(now - last_time).count();
      last_time = now;

      if (!ui_mode) {
        controller.Update(delta);
      }

      glm::quat rot = cube_transform->GetRotation();
      rot = glm::angleAxis(delta * 0.5f, glm::vec3(0.0f, 1.0f, 0.0f)) * rot;
      cube_transform->SetRotation(rot);

      scene.Update(delta);

      imgui_layer.BeginFrame();
      if (ui_mode) {
        ImGui::Begin("Material");
        ImGui::ColorEdit3("Base Color", &material.base_color.x);
        ImGui::SliderFloat("Metallic", &material.metallic, 0.0f, 1.0f);
        ImGui::SliderFloat("Roughness", &material.roughness, 0.04f, 1.0f);
        ImGui::End();

        ImGui::Begin("Camera");
        float speed = controller.GetMovementSpeed();
        if (ImGui::SliderFloat("Move speed", &speed, 0.5f, 20.0f)) {
          controller.SetMovementSpeed(speed);
        }
        float sens = controller.GetMouseSensitivity();
        if (ImGui::SliderFloat("Mouse sensitivity", &sens, 0.01f, 0.5f)) {
          controller.SetMouseSensitivity(sens);
        }
        float fov_deg = glm::degrees(camera->GetFov());
        if (ImGui::SliderFloat("FOV (deg)", &fov_deg, 30.0f, 110.0f)) {
          camera->SetFov(glm::radians(fov_deg));
        }
        glm::vec3 cam_pos_view = camera_transform->GetPosition();
        ImGui::Text("Position: (%.2f, %.2f, %.2f)",
                    cam_pos_view.x, cam_pos_view.y, cam_pos_view.z);
        ImGui::End();

        ImGui::Begin("Render");
        ImGui::SliderFloat("Exposure", &exposure, 0.1f, 5.0f);
        ImGui::SliderFloat("Gamma", &gamma, 1.0f, 3.0f);
        ImGui::Text("%.1f FPS (%.2f ms)", ImGui::GetIO().Framerate,
                    1000.0f / ImGui::GetIO().Framerate);
        ImGui::Checkbox("Show ImGui demo", &show_demo);
        ImGui::End();

        if (show_demo) {
          ImGui::ShowDemoWindow(&show_demo);
        }
      }
      imgui_layer.EndFrame();

      forward_pass->SetMaterial(material);

      glm::vec3 cam_pos = camera_transform->GetPosition();
      engine::UniformBufferObject ubo{
          .model = cube_transform->GetMatrix(),
          .view = camera->GetViewMatrix(),
          .proj = camera->GetProjectionMatrix(),
          .cam_pos = glm::vec4(cam_pos, 1.0f),
          .exposure = exposure,
          .gamma = gamma,
      };
      std::ranges::copy(light_positions, std::begin(ubo.light_positions));
      std::ranges::copy(light_colors, std::begin(ubo.light_colors));
      ubo_set.Update(renderer.GetCurrentFrame(), ubo);

      renderer.DrawFrame();
    }
    renderer.WaitIdle();

    event_bus.RemoveListener(&listener);
    event_bus.RemoveListener(&controller);
  }
};

}  // namespace app
