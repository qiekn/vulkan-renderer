module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

export module app;

import vulkan;
import std;
import engine.event;
import engine.window;
import engine.device;
import engine.swapchain;
import engine.pipeline;
import engine.renderer;
import engine.scene;
import engine.resource;

namespace app {

// ---------------------------------------------------------------------------: Constants

constexpr std::uint32_t kWindowWidth = 1280;
constexpr std::uint32_t kWindowHeight = 720;
constexpr std::string_view kWindowTitle = "Vulkan Engine";
constexpr std::string_view kTriangleShaderId = "triangle";
constexpr std::string_view kTriangleShaderPath = "assets/shaders/slang.spv";

// ---------------------------------------------------------------------------: Listener

// Demo listener: Esc closes, C dumps camera matrices, resize updates aspect.
class AppListener : public engine::EventListener {
 public:
  AppListener(engine::Window& window, engine::CameraComponent& camera)
      : window_(window), camera_(camera) {}

  void OnEvent(engine::Event& event) override {
    engine::EventDispatcher dispatcher(event);

    dispatcher.Dispatch<engine::KeyPressEvent>([this](engine::KeyPressEvent& e) {
      if (e.GetKeyCode() == GLFW_KEY_ESCAPE) {
        window_.RequestClose();
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
    auto triangle_shader = resources.Load<engine::ShaderResource>(
        std::string(kTriangleShaderId), device, std::filesystem::path(kTriangleShaderPath));
    if (!triangle_shader) {
      throw std::runtime_error("Failed to load triangle shader");
    }

    engine::Pipeline pipeline(device, *triangle_shader->GetModule(), swapchain.GetImageFormat());
    engine::Renderer renderer(window, device, swapchain, pipeline);

    engine::Scene scene;

    engine::Entity* triangle = scene.CreateEntity("Triangle");
    auto* triangle_transform = triangle->AddComponent<engine::TransformComponent>();

    engine::Entity* camera_entity = scene.CreateEntity("Camera");
    auto* camera_transform = camera_entity->AddComponent<engine::TransformComponent>();
    auto* camera = camera_entity->AddComponent<engine::CameraComponent>();
    camera_transform->SetPosition(glm::vec3(0.0f, 0.0f, 3.0f));
    auto extent = swapchain.GetExtent();
    camera->SetAspect(static_cast<float>(extent.width) / static_cast<float>(extent.height));

    AppListener listener(window, *camera);
    event_bus.AddListener(&listener);

    auto last_time = std::chrono::steady_clock::now();
    while (!window.ShouldClose()) {
      window.PollEvents();

      auto now = std::chrono::steady_clock::now();
      float delta = std::chrono::duration<float>(now - last_time).count();
      last_time = now;

      // Rotate the triangle entity around Y to show Update() is wired in.
      glm::quat rot = triangle_transform->GetRotation();
      rot = glm::angleAxis(delta, glm::vec3(0.0f, 1.0f, 0.0f)) * rot;
      triangle_transform->SetRotation(rot);

      scene.Update(delta);
      renderer.DrawFrame();
    }
    renderer.WaitIdle();

    event_bus.RemoveListener(&listener);
  }
};

}  // namespace app
