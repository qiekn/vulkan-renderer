module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

export module engine.window;

import vulkan;
import std;
import engine.event;

namespace engine {

// ---------------------------------------------------------------------------: Window

// GLFW wrapper. Publishes input/window events to the provided EventBus whenever
// GLFW fires a callback. The bus must outlive the Window.
export class Window {
public:
  Window(std::uint32_t width, std::uint32_t height, std::string_view title, EventBus& bus);
  ~Window();

  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;
  Window(Window&&) = delete;
  Window& operator=(Window&&) = delete;

  bool ShouldClose() const;
  void RequestClose() const;
  void PollEvents() const;
  void WaitEvents() const;
  void SetCursorDisabled(bool disabled);

  vk::Extent2D GetFramebufferSize() const;
  std::vector<const char*> GetRequiredInstanceExtensions() const;
  vk::raii::SurfaceKHR CreateSurface(const vk::raii::Instance& instance) const;

  bool WasResized() const { return resized_; }
  void ClearResized() { resized_ = false; }

private:
  static void FramebufferResizeCallback(GLFWwindow* window, int width, int height);
  static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
  static void CursorPosCallback(GLFWwindow* window, double x, double y);
  static void ScrollCallback(GLFWwindow* window, double x_offset, double y_offset);
  static void WindowCloseCallback(GLFWwindow* window);

  static Window* FromGLFW(GLFWwindow* window) {
    return static_cast<Window*>(glfwGetWindowUserPointer(window));
  }

  GLFWwindow* window_ = nullptr;
  EventBus* bus_ = nullptr;
  bool resized_ = false;
};

// ---------------------------------------------------------------------------: Implementation

Window::Window(std::uint32_t width, std::uint32_t height, std::string_view title, EventBus& bus)
    : bus_(&bus) {
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
  glfwSetKeyCallback(window_, &KeyCallback);
  glfwSetMouseButtonCallback(window_, &MouseButtonCallback);
  glfwSetCursorPosCallback(window_, &CursorPosCallback);
  glfwSetScrollCallback(window_, &ScrollCallback);
  glfwSetWindowCloseCallback(window_, &WindowCloseCallback);
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

void Window::RequestClose() const {
  glfwSetWindowShouldClose(window_, GLFW_TRUE);
}

void Window::PollEvents() const {
  glfwPollEvents();
}

void Window::WaitEvents() const {
  glfwWaitEvents();
}

void Window::SetCursorDisabled(bool disabled) {
  glfwSetInputMode(window_, GLFW_CURSOR,
                   disabled ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
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

// ---------------------------------------------------------------------------: GLFW callbacks

void Window::FramebufferResizeCallback(GLFWwindow* window, int width, int height) {
  auto* self = FromGLFW(window);
  if (self == nullptr) return;
  self->resized_ = true;
  WindowResizeEvent event(static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height));
  self->bus_->Publish(event);
}

void Window::KeyCallback(GLFWwindow* window, int key, int /*scancode*/, int action, int mods) {
  auto* self = FromGLFW(window);
  if (self == nullptr) return;
  if (action == GLFW_PRESS || action == GLFW_REPEAT) {
    KeyPressEvent event(key, mods, action == GLFW_REPEAT);
    self->bus_->Publish(event);
  } else if (action == GLFW_RELEASE) {
    KeyReleaseEvent event(key, mods);
    self->bus_->Publish(event);
  }
}

void Window::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  auto* self = FromGLFW(window);
  if (self == nullptr) return;
  if (action == GLFW_PRESS) {
    MouseButtonPressEvent event(button, mods);
    self->bus_->Publish(event);
  } else if (action == GLFW_RELEASE) {
    MouseButtonReleaseEvent event(button, mods);
    self->bus_->Publish(event);
  }
}

void Window::CursorPosCallback(GLFWwindow* window, double x, double y) {
  auto* self = FromGLFW(window);
  if (self == nullptr) return;
  MouseMoveEvent event(x, y);
  self->bus_->Publish(event);
}

void Window::ScrollCallback(GLFWwindow* window, double x_offset, double y_offset) {
  auto* self = FromGLFW(window);
  if (self == nullptr) return;
  MouseScrollEvent event(x_offset, y_offset);
  self->bus_->Publish(event);
}

void Window::WindowCloseCallback(GLFWwindow* window) {
  auto* self = FromGLFW(window);
  if (self == nullptr) return;
  WindowCloseEvent event;
  self->bus_->Publish(event);
}

}  // namespace engine
