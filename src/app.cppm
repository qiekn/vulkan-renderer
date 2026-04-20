module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <imgui.h>

export module app;

import vulkan;
import std;
import engine.event;
import engine.window;
import engine.device;
import engine.swapchain;
import engine.model;
import engine.model_loader;
import engine.material_bindings;
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
import engine.audio;

namespace app {

// ---------------------------------------------------------------------------: Constants

constexpr std::uint32_t kWindowWidth = 1280;
constexpr std::uint32_t kWindowHeight = 720;
constexpr std::string_view kWindowTitle = "Vulkan Engine";
constexpr std::string_view kPbrShaderId = "pbr";
constexpr std::string_view kPbrShaderPath = "assets/shaders/slang.spv";
constexpr std::string_view kDefaultModelPath = "assets/models/damaged_helmet/DamagedHelmet.glb";
constexpr std::string_view kAmbientClipId = "ambient";
constexpr std::string_view kAmbientClipPath = "assets/sounds/ambient.ogg";
constexpr std::string_view kPingClipId = "ping";
constexpr std::string_view kPingClipPath = "assets/sounds/ping.wav";

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
    // AudioSystem is intentionally declared before Scene: stack destruction is
    // reverse order, so Scene (and its AudioSourceComponents) must tear down
    // first — before the AudioSystem uninit's the ma_sounds they point at.
    engine::AudioSystem audio;
    audio.LoadClip(std::string(kAmbientClipId), std::filesystem::path(kAmbientClipPath));
    audio.LoadClip(std::string(kPingClipId), std::filesystem::path(kPingClipPath));

    engine::Swapchain swapchain(window, device);

    engine::ResourceManager resources;
    auto pbr_shader = resources.Load<engine::ShaderResource>(
        std::string(kPbrShaderId), device, std::filesystem::path(kPbrShaderPath));
    if (!pbr_shader) {
      throw std::runtime_error("Failed to load pbr shader");
    }

    // Load the glTF scene first — the pipeline's descriptor set layout for
    // set=1 comes from MaterialBindings, which needs the Model to know how
    // many materials to allocate descriptor sets for.
    engine::Model model = engine::LoadGltfModel(device, std::filesystem::path(kDefaultModelPath));
    std::cout << "[model] loaded " << kDefaultModelPath
              << " nodes=" << model.nodes.size()
              << " materials=" << model.materials.size()
              << " textures=" << model.textures.size()
              << " animations=" << model.animations.size() << '\n';
    engine::MaterialBindings material_bindings(device, model);

    engine::UniformBufferSet ubo_set(device, engine::Renderer::kMaxFramesInFlight);

    auto vertex_attributes = engine::ModelVertex::GetAttributeDescriptions();
    vk::PushConstantRange draw_range{
        .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(engine::DrawPushConstants),
    };
    std::array descriptor_layouts = {
        *ubo_set.GetLayout(),
        *material_bindings.GetLayout(),
    };
    engine::PipelineConfig pipeline_config{
        .shader_module = *pbr_shader->GetModule(),
        .color_format = swapchain.GetImageFormat(),
        .depth_format = swapchain.GetDepthFormat(),
        .vertex_binding = engine::ModelVertex::GetBindingDescription(),
        .vertex_attributes = std::span(vertex_attributes),
        .descriptor_set_layouts = std::span(descriptor_layouts),
        .push_constant_ranges = std::span(&draw_range, 1),
        .cull_mode = vk::CullModeFlagBits::eBack,
        .front_face = vk::FrontFace::eCounterClockwise,
    };
    engine::Pipeline pipeline(device, pipeline_config);

    engine::RenderPassManager pass_manager;
    auto* forward_pass =
        pass_manager.AddPass<engine::ForwardPass>(pipeline, ubo_set, model, material_bindings);
    pass_manager.AddPass<engine::ImGuiPass>();

    // ImGuiLayer builds its graphics pipeline against the swapchain color
    // format, so it has to be alive before Renderer records its first frame.
    engine::ImGuiLayer imgui_layer(window, device, swapchain);

    engine::Renderer renderer(window, device, swapchain, pass_manager);

    engine::Scene scene;

    engine::Entity* camera_entity = scene.CreateEntity("Camera");
    auto* camera_transform = camera_entity->AddComponent<engine::TransformComponent>();
    auto* camera = camera_entity->AddComponent<engine::CameraComponent>();
    camera_entity->AddComponent<engine::AudioListenerComponent>(&audio.GetListener());
    camera_transform->SetPosition(glm::vec3(0.0f, 0.0f, 3.5f));
    auto extent = swapchain.GetExtent();
    camera->SetAspect(static_cast<float>(extent.width) / static_cast<float>(extent.height));

    engine::CameraController controller(*camera_transform, *camera);
    event_bus.AddListener(&controller);

    bool ui_mode = false;
    AppListener listener(window, *camera, controller, ui_mode);
    event_bus.AddListener(&listener);

    window.SetCursorDisabled(true);

    // Four point lights in a loose ring around the model so every face picks up
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
    bool animate_rotation = true;
    bool play_animation = true;
    int active_animation = 0;
    float master_volume = 0.5f;
    audio.SetMasterVolume(master_volume);

    // Ambient loop source — parked at origin so spatial attenuation varies as
    // the camera moves around the scene. If the clip wasn't found (missing
    // asset), CreateSource returns nullptr and the entity stays silent.
    engine::Entity* ambient_entity = scene.CreateEntity("AmbientAudio");
    ambient_entity->AddComponent<engine::TransformComponent>()
        ->SetPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    if (auto* ambient_src = audio.CreateSource(std::string(kAmbientClipId))) {
      ambient_src->SetLooping(true);
      ambient_src->Play();
      ambient_entity->AddComponent<engine::AudioSourceComponent>(ambient_src);
    }

    // Flat list of object instances — tutorial-style layout. Each entry becomes
    // one world-space model matrix per frame; ForwardPass walks the scene graph
    // once per matrix. `animation_angle` is shared so every helmet rotates in
    // lockstep; swap to per-instance phases if you want a livelier scene.
    struct InstanceTransform {
      glm::vec3 position;
      glm::vec3 rotation_euler_deg;
      glm::vec3 scale;
    };
    const std::array<InstanceTransform, 10> kInstanceTransforms = {{
        {glm::vec3( 0.0f,  0.0f,  0.0f), glm::vec3(  0.0f,   0.0f, 0.0f), glm::vec3(1.0f)},
        {glm::vec3(-2.5f,  0.0f, -1.0f), glm::vec3(  0.0f,  45.0f, 0.0f), glm::vec3(0.8f)},
        {glm::vec3( 2.5f,  0.0f, -1.0f), glm::vec3(  0.0f, -45.0f, 0.0f), glm::vec3(0.8f)},
        {glm::vec3(-2.0f,  0.0f, -3.5f), glm::vec3(  0.0f,  30.0f, 0.0f), glm::vec3(0.7f)},
        {glm::vec3( 2.0f,  0.0f, -3.5f), glm::vec3(  0.0f, -30.0f, 0.0f), glm::vec3(0.7f)},
        {glm::vec3(-2.0f,  0.0f,  2.0f), glm::vec3(  0.0f, -30.0f, 0.0f), glm::vec3(0.6f)},
        {glm::vec3( 2.0f,  0.0f,  2.0f), glm::vec3(  0.0f,  30.0f, 0.0f), glm::vec3(0.6f)},
        {glm::vec3( 0.0f,  2.5f, -2.5f), glm::vec3( 45.0f,   0.0f, 0.0f), glm::vec3(0.5f)},
        {glm::vec3( 0.0f, -1.5f, -2.5f), glm::vec3(-30.0f,   0.0f, 0.0f), glm::vec3(0.5f)},
        {glm::vec3( 0.0f,  0.5f, -5.5f), glm::vec3(  0.0f, 180.0f, 0.0f), glm::vec3(1.2f)},
    }};
    int visible_instance_count = static_cast<int>(kInstanceTransforms.size());
    float animation_angle = 0.0f;
    std::vector<glm::mat4> instance_matrices;
    instance_matrices.reserve(kInstanceTransforms.size());

    auto last_time = std::chrono::steady_clock::now();
    while (!window.ShouldClose()) {
      window.PollEvents();

      auto now = std::chrono::steady_clock::now();
      float delta = std::chrono::duration<float>(now - last_time).count();
      last_time = now;

      if (!ui_mode) {
        controller.Update(delta);
      }

      if (animate_rotation) {
        animation_angle += delta * 0.5f;
      }

      if (play_animation && !model.animations.empty()) {
        const std::uint32_t anim_count = static_cast<std::uint32_t>(model.animations.size());
        const std::uint32_t clamped =
            std::min<std::uint32_t>(static_cast<std::uint32_t>(active_animation), anim_count - 1);
        model.UpdateAnimation(clamped, delta);
      }

      scene.Update(delta);
      audio.Update(delta);

      imgui_layer.BeginFrame();
      if (ui_mode) {
        ImGui::Begin("Model");
        ImGui::Checkbox("Auto-rotate", &animate_rotation);
        ImGui::SliderInt("Instances", &visible_instance_count, 1,
                         static_cast<int>(kInstanceTransforms.size()));
        ImGui::Text("Nodes: %zu, Materials: %zu, Textures: %zu",
                    model.nodes.size(), model.materials.size(), model.textures.size());
        if (model.animations.empty()) {
          ImGui::TextDisabled("Animations: none");
        } else {
          ImGui::Checkbox("Play animation", &play_animation);
          if (model.animations.size() > 1) {
            ImGui::SliderInt("Animation", &active_animation, 0,
                             static_cast<int>(model.animations.size()) - 1);
          }
          const auto& anim =
              model.animations[std::min<std::size_t>(active_animation, model.animations.size() - 1)];
          ImGui::Text("%s: %.2fs / %.2fs", anim.name.empty() ? "anim" : anim.name.c_str(),
                      anim.current_time, anim.end);
        }
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

        ImGui::Begin("Audio");
        if (ImGui::SliderFloat("Master volume", &master_volume, 0.0f, 1.0f)) {
          audio.SetMasterVolume(master_volume);
        }
        if (ImGui::Button("Play Ping")) {
          audio.PlayOneShot(std::filesystem::path(kPingClipPath));
        }
        glm::vec3 listener_pos = camera_transform->GetPosition();
        ImGui::Text("Listener: (%.2f, %.2f, %.2f)",
                    listener_pos.x, listener_pos.y, listener_pos.z);
        ImGui::End();

        if (show_demo) {
          ImGui::ShowDemoWindow(&show_demo);
        }
      }
      imgui_layer.EndFrame();

      instance_matrices.clear();
      const int count = std::min<int>(visible_instance_count,
                                      static_cast<int>(kInstanceTransforms.size()));
      for (int i = 0; i < count; ++i) {
        const auto& t = kInstanceTransforms[i];
        glm::mat4 m(1.0f);
        m = glm::translate(m, t.position);
        m = glm::rotate(m, animation_angle, glm::vec3(0.0f, 1.0f, 0.0f));
        m = glm::rotate(m, glm::radians(t.rotation_euler_deg.x), glm::vec3(1.0f, 0.0f, 0.0f));
        m = glm::rotate(m, glm::radians(t.rotation_euler_deg.y), glm::vec3(0.0f, 1.0f, 0.0f));
        m = glm::rotate(m, glm::radians(t.rotation_euler_deg.z), glm::vec3(0.0f, 0.0f, 1.0f));
        m = glm::scale(m, t.scale);
        instance_matrices.push_back(m);
      }
      forward_pass->SetInstances(std::span(instance_matrices));

      glm::vec3 cam_pos = camera_transform->GetPosition();
      engine::UniformBufferObject ubo{
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
