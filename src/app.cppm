module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

export module app;

import vulkan;
import std;
import engine.event;
import engine.window;
import engine.device;
import engine.swapchain;
import engine.pipeline;
import engine.renderer;

namespace app {

// ---------------------------------------------------------------------------: Constants

constexpr std::uint32_t kWindowWidth = 1280;
constexpr std::uint32_t kWindowHeight = 720;
constexpr std::string_view kWindowTitle = "Vulkan Engine";
constexpr std::string_view kShaderPath = "assets/shaders/slang.spv";

// ---------------------------------------------------------------------------: Input listener

// Demo listener that routes input events: Escape closes the window, other
// keys log once. Also reports framebuffer resizes. Shows off type-safe
// dispatch via engine::EventDispatcher.
class InputListener : public engine::EventListener {
 public:
  explicit InputListener(engine::Window& window) : window_(window) {}

  void OnEvent(engine::Event& event) override {
    engine::EventDispatcher dispatcher(event);
    dispatcher.Dispatch<engine::KeyPressEvent>([this](engine::KeyPressEvent& e) {
      if (e.GetKeyCode() == GLFW_KEY_ESCAPE) {
        window_.RequestClose();
        return true;
      }
      if (!e.IsRepeat()) {
        std::cout << "[input] key=" << e.GetKeyCode() << " mods=" << e.GetMods() << '\n';
      }
      return false;
    });
    dispatcher.Dispatch<engine::WindowResizeEvent>([](engine::WindowResizeEvent& e) {
      std::cout << "[input] resize=" << e.GetWidth() << 'x' << e.GetHeight() << '\n';
      return false;
    });
  }

 private:
  engine::Window& window_;
};

// ---------------------------------------------------------------------------: Application

export class Application {
 public:
  void Run() {
    engine::EventBus event_bus;
    engine::Window window(kWindowWidth, kWindowHeight, kWindowTitle, event_bus);
    engine::Device device(window);
    engine::Swapchain swapchain(window, device);
    engine::Pipeline pipeline(device, kShaderPath, swapchain.GetImageFormat());
    engine::Renderer renderer(window, device, swapchain, pipeline);

    InputListener input_listener(window);
    event_bus.AddListener(&input_listener);

    while (!window.ShouldClose()) {
      window.PollEvents();
      renderer.DrawFrame();
    }
    renderer.WaitIdle();

    event_bus.RemoveListener(&input_listener);
  }
};

}  // namespace app
