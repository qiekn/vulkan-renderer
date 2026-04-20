module;

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

export module engine.imgui_layer;

import vulkan;
import std;
import engine.device;
import engine.swapchain;
import engine.window;

namespace engine {

// ---------------------------------------------------------------------------: Theme

// Ported from ../opengl-lab/core/core/imgui_layer.cc — neutral dark palette
// that matches the sibling project's look so both engines feel the same.
namespace {

void ApplyDarkThemePalette() {
  auto& colors = ImGui::GetStyle().Colors;
  colors[ImGuiCol_WindowBg] = ImVec4{0.10f, 0.105f, 0.11f, 1.00f};

  colors[ImGuiCol_Header] = ImVec4{0.20f, 0.205f, 0.21f, 1.00f};
  colors[ImGuiCol_HeaderHovered] = ImVec4{0.30f, 0.305f, 0.31f, 1.00f};
  colors[ImGuiCol_HeaderActive] = ImVec4{0.15f, 0.1505f, 0.151f, 1.00f};

  colors[ImGuiCol_Button] = ImVec4{0.20f, 0.205f, 0.21f, 1.00f};
  colors[ImGuiCol_ButtonHovered] = ImVec4{0.30f, 0.305f, 0.31f, 1.00f};
  colors[ImGuiCol_ButtonActive] = ImVec4{0.15f, 0.1505f, 0.151f, 1.00f};

  colors[ImGuiCol_FrameBg] = ImVec4{0.20f, 0.205f, 0.21f, 1.00f};
  colors[ImGuiCol_FrameBgHovered] = ImVec4{0.30f, 0.305f, 0.31f, 1.00f};
  colors[ImGuiCol_FrameBgActive] = ImVec4{0.15f, 0.1505f, 0.151f, 1.00f};

  colors[ImGuiCol_Tab] = ImVec4{0.15f, 0.1505f, 0.151f, 1.00f};
  colors[ImGuiCol_TabHovered] = ImVec4{0.38f, 0.3805f, 0.381f, 1.00f};
  colors[ImGuiCol_TabActive] = ImVec4{0.28f, 0.2805f, 0.281f, 1.00f};
  colors[ImGuiCol_TabUnfocused] = ImVec4{0.15f, 0.1505f, 0.151f, 1.00f};
  colors[ImGuiCol_TabUnfocusedActive] = ImVec4{0.20f, 0.205f, 0.21f, 1.00f};

  colors[ImGuiCol_TitleBg] = ImVec4{0.15f, 0.1505f, 0.151f, 1.00f};
  colors[ImGuiCol_TitleBgActive] = ImVec4{0.15f, 0.1505f, 0.151f, 1.00f};
}

}  // namespace

// ---------------------------------------------------------------------------: ImGuiLayer

// Owns the Dear ImGui context + the glfw/vulkan backends. Keeps a dedicated
// descriptor pool so ImGui's font atlas and any user-added textures do not
// share the engine-global pool. Uses dynamic rendering: the pipeline is built
// against a VkPipelineRenderingCreateInfo with the swapchain's color format,
// which is the same layout ForwardPass and the Renderer agree on.
export class ImGuiLayer {
public:
  ImGuiLayer(Window& window, Device& device, Swapchain& swapchain);
  ~ImGuiLayer();

  ImGuiLayer(const ImGuiLayer&) = delete;
  ImGuiLayer& operator=(const ImGuiLayer&) = delete;
  ImGuiLayer(ImGuiLayer&&) = delete;
  ImGuiLayer& operator=(ImGuiLayer&&) = delete;

  void BeginFrame();
  void EndFrame();

  bool WantCaptureMouse() const { return ImGui::GetIO().WantCaptureMouse; }
  bool WantCaptureKeyboard() const { return ImGui::GetIO().WantCaptureKeyboard; }

private:
  Device& device_;
  vk::raii::DescriptorPool pool_ = nullptr;
  VkFormat color_format_ = VK_FORMAT_UNDEFINED;
};

// ---------------------------------------------------------------------------: Implementation

ImGuiLayer::ImGuiLayer(Window& window, Device& device, Swapchain& swapchain)
    : device_(device) {
  // One combined-image-sampler descriptor is enough for the font atlas. Any
  // user-registered textures (ImGui_ImplVulkan_AddTexture) would need more —
  // bump maxSets / descriptorCount then.
  vk::DescriptorPoolSize pool_size{
      .type = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = 8,
  };
  vk::DescriptorPoolCreateInfo pool_info{
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = 8,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
  };
  pool_ = vk::raii::DescriptorPool(device_.GetLogicalDevice(), pool_info);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  // Don't leave an imgui.ini next to the exe — keeps the working dir clean.
  io.IniFilename = nullptr;

  // DPI-scaled fonts + rounded style + custom dark palette, mirrored from
  // ../opengl-lab so the two engines have the same look & feel. The TTFs are
  // copied into assets/fonts/opensans/ and synced to build/ by sync_assets.
  const float scale = ImGui_ImplGlfw_GetContentScaleForWindow(window.GetGlfwHandle());
  const float font_size = 18.0f * scale;
  io.Fonts->AddFontFromFileTTF("assets/fonts/opensans/OpenSans-Bold.ttf", font_size);
  if (ImFont* regular = io.Fonts->AddFontFromFileTTF(
          "assets/fonts/opensans/OpenSans-Regular.ttf", font_size)) {
    io.FontDefault = regular;
  }

  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 8.0f;
  style.FrameRounding = 8.0f;
  style.ScaleAllSizes(scale);
  ApplyDarkThemePalette();

  // install_callbacks=true: ImGui chains onto the GLFW callbacks that
  // engine::Window already installed, so KeyPressEvent/MouseMoveEvent etc.
  // still flow through the bus while ImGui also sees the input.
  ImGui_ImplGlfw_InitForVulkan(window.GetGlfwHandle(), true);

  color_format_ = static_cast<VkFormat>(swapchain.GetImageFormat());
  VkPipelineRenderingCreateInfoKHR rendering_info{};
  rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR;
  rendering_info.colorAttachmentCount = 1;
  rendering_info.pColorAttachmentFormats = &color_format_;

  ImGui_ImplVulkan_InitInfo init_info{};
  init_info.ApiVersion = VK_API_VERSION_1_3;
  init_info.Instance = *device_.GetInstance();
  init_info.PhysicalDevice = *device_.GetPhysicalDevice();
  init_info.Device = *device_.GetLogicalDevice();
  init_info.QueueFamily = device_.GetGraphicsQueueFamily();
  init_info.Queue = *device_.GetGraphicsQueue();
  init_info.DescriptorPool = *pool_;
  init_info.MinImageCount = swapchain.GetImageCount();
  init_info.ImageCount = swapchain.GetImageCount();
  init_info.UseDynamicRendering = true;
  init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
  init_info.PipelineInfoMain.PipelineRenderingCreateInfo = rendering_info;
  init_info.CheckVkResultFn = [](VkResult result) {
    if (result != VK_SUCCESS) {
      std::cerr << "[imgui vk] VkResult=" << static_cast<int>(result) << '\n';
    }
  };

  if (!ImGui_ImplVulkan_Init(&init_info)) {
    throw std::runtime_error("Failed to initialize ImGui Vulkan backend");
  }
}

ImGuiLayer::~ImGuiLayer() {
  device_.GetLogicalDevice().waitIdle();
  ImGui_ImplVulkan_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
}

void ImGuiLayer::BeginFrame() {
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
}

void ImGuiLayer::EndFrame() {
  ImGui::Render();
}

}  // namespace engine
