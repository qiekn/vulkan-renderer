export module app;

import vulkan;
import std;
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

// ---------------------------------------------------------------------------: Application

export class Application {
 public:
  void Run() {
    engine::Window window(kWindowWidth, kWindowHeight, kWindowTitle);
    engine::Device device(window);
    engine::Swapchain swapchain(window, device);
    engine::Pipeline pipeline(device, kShaderPath, swapchain.GetImageFormat());
    engine::Renderer renderer(window, device, swapchain, pipeline);

    while (!window.ShouldClose()) {
      window.PollEvents();
      renderer.DrawFrame();
    }
    renderer.WaitIdle();
  }
};

}  // namespace app
