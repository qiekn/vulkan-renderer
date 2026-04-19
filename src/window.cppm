module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

export module engine.window;

import vulkan;
import std;

namespace engine {

// ---------------------------------------------------------------------------: Window

export class Window {
 public:
  Window(std::uint32_t width, std::uint32_t height, std::string_view title);
  ~Window();

  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;
  Window(Window&&) = delete;
  Window& operator=(Window&&) = delete;

  bool ShouldClose() const;
  void PollEvents() const;
  void WaitEvents() const;

  vk::Extent2D GetFramebufferSize() const;
  std::vector<const char*> GetRequiredInstanceExtensions() const;
  vk::raii::SurfaceKHR CreateSurface(const vk::raii::Instance& instance) const;

  bool WasResized() const { return resized_; }
  void ClearResized() { resized_ = false; }

 private:
  static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);

  GLFWwindow* window_ = nullptr;
  bool resized_ = false;
};

// ---------------------------------------------------------------------------: Implementation

Window::Window(std::uint32_t width, std::uint32_t height, std::string_view title) {
  if (glfwInit() == GLFW_FALSE) {
    throw std::runtime_error("Failed to initialize GLFW");
  }
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  window_ = glfwCreateWindow(static_cast<int>(width), static_cast<int>(height),
                             std::string(title).c_str(), nullptr, nullptr);
  if (window_ == nullptr) {
    glfwTerminate();
    throw std::runtime_error("Failed to create GLFW window");
  }
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, &FramebufferResizeCallback);
}

Window::~Window() {
  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
  }
  glfwTerminate();
}

bool Window::ShouldClose() const {
  return glfwWindowShouldClose(window_) == GLFW_TRUE;
}

void Window::PollEvents() const {
  glfwPollEvents();
}

void Window::WaitEvents() const {
  glfwWaitEvents();
}

vk::Extent2D Window::GetFramebufferSize() const {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  return vk::Extent2D{static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height)};
}

std::vector<const char*> Window::GetRequiredInstanceExtensions() const {
  std::uint32_t count = 0;
  const char** extensions = glfwGetRequiredInstanceExtensions(&count);
  return {extensions, extensions + count};
}

vk::raii::SurfaceKHR Window::CreateSurface(const vk::raii::Instance& instance) const {
  VkSurfaceKHR raw_surface = VK_NULL_HANDLE;
  if (glfwCreateWindowSurface(*instance, window_, nullptr, &raw_surface) != VK_SUCCESS) {
    throw std::runtime_error("Failed to create window surface");
  }
  return vk::raii::SurfaceKHR(instance, raw_surface);
}

void Window::FramebufferResizeCallback(GLFWwindow* window, int /*width*/, int /*height*/) {
  auto* self = static_cast<Window*>(glfwGetWindowUserPointer(window));
  if (self != nullptr) {
    self->resized_ = true;
  }
}

}  // namespace engine
