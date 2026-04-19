module;

#include <stdexcept>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

export module app;

import vulkan;
import std;

namespace app {

// -----------------------------------------------------------------------------: Constants

constexpr std::uint32_t kWindowWidth = 1280;
constexpr std::uint32_t kWindowHeight = 720;
constexpr const char* kWindowTitle = "Vulkan Engine";

#ifdef NDEBUG
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

// -----------------------------------------------------------------------------: Application

export class Application {
 public:
  Application() = default;
  ~Application() { Cleanup(); }

  Application(const Application&) = delete;
  Application& operator=(const Application&) = delete;
  Application(Application&&) = delete;
  Application& operator=(Application&&) = delete;

  void Run() {
    InitWindow();
    InitVulkan();
    MainLoop();
  }

 private:
  void InitWindow() {
    if (glfwInit() == GLFW_FALSE) {
      throw std::runtime_error("failed to initialize GLFW");
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(kWindowWidth, kWindowHeight, kWindowTitle, nullptr, nullptr);
    if (window_ == nullptr) {
      throw std::runtime_error("failed to create GLFW window");
    }
  }

  void InitVulkan() {
    // TODO(engine): instance / device / swapchain / dynamic rendering setup
    // will be added following the "Engine Architecture" chapter.
  }

  void MainLoop() {
    while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
      glfwPollEvents();
    }
  }

  void Cleanup() {
    if (window_ != nullptr) {
      glfwDestroyWindow(window_);
      window_ = nullptr;
    }
    glfwTerminate();
  }

  GLFWwindow* window_ = nullptr;
};

}  // namespace app
