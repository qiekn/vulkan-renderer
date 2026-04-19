module;

#include <cassert>

export module engine.device;

import vulkan;
import std;
import engine.window;

namespace engine {

// ---------------------------------------------------------------------------: Constants

#ifdef NDEBUG
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

inline constexpr std::array kValidationLayers = {
    "VK_LAYER_KHRONOS_validation",
};

inline constexpr std::array kRequiredDeviceExtensions = {
    vk::KHRSwapchainExtensionName,
};

// ---------------------------------------------------------------------------: Device

// Bundles instance / surface / physical device / logical device / graphics queue.
// A single engine::Device owns everything needed to construct higher-level
// Vulkan objects (swapchain, pipelines, command pools, ...).
export class Device {
 public:
  explicit Device(Window& window);

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;

  const vk::raii::Context& GetContext() const { return context_; }
  const vk::raii::Instance& GetInstance() const { return instance_; }
  const vk::raii::SurfaceKHR& GetSurface() const { return surface_; }
  const vk::raii::PhysicalDevice& GetPhysicalDevice() const { return physical_device_; }
  const vk::raii::Device& GetLogicalDevice() const { return device_; }
  const vk::raii::Queue& GetGraphicsQueue() const { return graphics_queue_; }
  std::uint32_t GetGraphicsQueueFamily() const { return graphics_queue_family_; }

 private:
  void CreateInstance();
  void SetupDebugMessenger();
  void PickPhysicalDevice();
  bool IsDeviceSuitable(const vk::raii::PhysicalDevice& pd) const;
  std::uint32_t FindGraphicsQueueFamily() const;
  void CreateLogicalDevice();

  std::vector<const char*> GetRequiredInstanceExtensions() const;
  void CheckValidationLayerSupport() const;

  static vk::Bool32 DebugCallback(
      vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
      vk::DebugUtilsMessageTypeFlagsEXT type,
      const vk::DebugUtilsMessengerCallbackDataEXT* data,
      void* user_data);

  Window& window_;

  vk::raii::Context context_;
  vk::raii::Instance instance_ = nullptr;
  vk::raii::DebugUtilsMessengerEXT debug_messenger_ = nullptr;
  vk::raii::SurfaceKHR surface_ = nullptr;
  vk::raii::PhysicalDevice physical_device_ = nullptr;
  vk::raii::Device device_ = nullptr;
  vk::raii::Queue graphics_queue_ = nullptr;
  std::uint32_t graphics_queue_family_ = 0;
};

// ---------------------------------------------------------------------------: Implementation

Device::Device(Window& window) : window_(window) {
  CreateInstance();
  SetupDebugMessenger();
  surface_ = window_.CreateSurface(instance_);
  PickPhysicalDevice();
  CreateLogicalDevice();
}

void Device::CreateInstance() {
  if (kEnableValidationLayers) {
    CheckValidationLayerSupport();
  }

  constexpr vk::ApplicationInfo app_info{
      .pApplicationName = "Vulkan Engine",
      .applicationVersion = vk::makeApiVersion(0, 0, 1, 0),
      .pEngineName = "vulkan-engine",
      .engineVersion = vk::makeApiVersion(0, 0, 1, 0),
      .apiVersion = vk::ApiVersion14,
  };

  auto required_extensions = GetRequiredInstanceExtensions();

  vk::InstanceCreateInfo create_info{
      .pApplicationInfo = &app_info,
      .enabledLayerCount = kEnableValidationLayers ? static_cast<std::uint32_t>(kValidationLayers.size()) : 0,
      .ppEnabledLayerNames = kEnableValidationLayers ? kValidationLayers.data() : nullptr,
      .enabledExtensionCount = static_cast<std::uint32_t>(required_extensions.size()),
      .ppEnabledExtensionNames = required_extensions.data(),
  };

  instance_ = vk::raii::Instance(context_, create_info);
}

void Device::CheckValidationLayerSupport() const {
  auto available_layers = context_.enumerateInstanceLayerProperties();
  for (const char* layer : kValidationLayers) {
    bool found = std::ranges::any_of(available_layers, [layer](const auto& prop) {
      return std::strcmp(prop.layerName, layer) == 0;
    });
    if (!found) {
      throw std::runtime_error("Validation layer not available: " + std::string(layer));
    }
  }
}

std::vector<const char*> Device::GetRequiredInstanceExtensions() const {
  auto extensions = window_.GetRequiredInstanceExtensions();
  if (kEnableValidationLayers) {
    extensions.push_back(vk::EXTDebugUtilsExtensionName);
  }
  return extensions;
}

void Device::SetupDebugMessenger() {
  if (!kEnableValidationLayers) {
    return;
  }

  vk::DebugUtilsMessengerCreateInfoEXT create_info{
      .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning
                       | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
      .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral
                   | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
                   | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
      .pfnUserCallback = &DebugCallback,
  };
  debug_messenger_ = instance_.createDebugUtilsMessengerEXT(create_info);
}

vk::Bool32 Device::DebugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT /*type*/,
    const vk::DebugUtilsMessengerCallbackDataEXT* data,
    void* /*user_data*/) {
  if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    std::cerr << "[validation] " << data->pMessage << '\n';
  }
  return vk::False;
}

void Device::PickPhysicalDevice() {
  auto physicals = instance_.enumeratePhysicalDevices();
  auto it = std::ranges::find_if(physicals, [this](const auto& pd) { return IsDeviceSuitable(pd); });
  if (it == physicals.end()) {
    throw std::runtime_error("Failed to find a suitable GPU");
  }
  physical_device_ = *it;
}

bool Device::IsDeviceSuitable(const vk::raii::PhysicalDevice& pd) const {
  bool supports_vulkan_1_3 = pd.getProperties().apiVersion >= vk::ApiVersion13;

  auto queue_families = pd.getQueueFamilyProperties();
  bool has_graphics = std::ranges::any_of(queue_families, [](const auto& qfp) {
    return static_cast<bool>(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
  });

  auto available_extensions = pd.enumerateDeviceExtensionProperties();
  bool has_required_extensions = std::ranges::all_of(kRequiredDeviceExtensions, [&](const char* required) {
    return std::ranges::any_of(available_extensions, [required](const auto& ext) {
      return std::strcmp(ext.extensionName, required) == 0;
    });
  });

  auto feats = pd.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                                vk::PhysicalDeviceVulkan13Features>();
  bool has_required_features = feats.get<vk::PhysicalDeviceVulkan11Features>().shaderDrawParameters
                             && feats.get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering
                             && feats.get<vk::PhysicalDeviceVulkan13Features>().synchronization2;

  return supports_vulkan_1_3 && has_graphics && has_required_extensions && has_required_features;
}

std::uint32_t Device::FindGraphicsQueueFamily() const {
  auto queue_families = physical_device_.getQueueFamilyProperties();
  for (std::uint32_t i = 0; i < queue_families.size(); ++i) {
    bool has_graphics = static_cast<bool>(queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics);
    bool has_present = physical_device_.getSurfaceSupportKHR(i, *surface_) == vk::True;
    if (has_graphics && has_present) {
      return i;
    }
  }
  throw std::runtime_error("No queue family with graphics + present support");
}

void Device::CreateLogicalDevice() {
  graphics_queue_family_ = FindGraphicsQueueFamily();

  float queue_priority = 1.0f;
  vk::DeviceQueueCreateInfo queue_create_info{
      .queueFamilyIndex = graphics_queue_family_,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
  };

  vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features,
                     vk::PhysicalDeviceVulkan13Features>
      feature_chain = {
          vk::PhysicalDeviceFeatures2{},
          vk::PhysicalDeviceVulkan11Features{.shaderDrawParameters = vk::True},
          vk::PhysicalDeviceVulkan13Features{.synchronization2 = vk::True, .dynamicRendering = vk::True},
      };

  vk::DeviceCreateInfo create_info{
      .pNext = &feature_chain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_create_info,
      .enabledExtensionCount = static_cast<std::uint32_t>(kRequiredDeviceExtensions.size()),
      .ppEnabledExtensionNames = kRequiredDeviceExtensions.data(),
  };

  device_ = vk::raii::Device(physical_device_, create_info);
  graphics_queue_ = vk::raii::Queue(device_, graphics_queue_family_, 0);
}

}  // namespace engine
