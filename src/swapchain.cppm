export module engine.swapchain;

import vulkan;
import std;
import engine.window;
import engine.device;

namespace engine {

// ---------------------------------------------------------------------------: Swapchain

// Wraps vk::raii::SwapchainKHR + its image views. Recreate() tears down and
// rebuilds, so it is safe to call on framebuffer resize.
export class Swapchain {
public:
  Swapchain(Window& window, Device& device);

  Swapchain(const Swapchain&) = delete;
  Swapchain& operator=(const Swapchain&) = delete;
  Swapchain(Swapchain&&) = delete;
  Swapchain& operator=(Swapchain&&) = delete;

  void Recreate();

  const vk::raii::SwapchainKHR& GetHandle() const { return swapchain_; }
  const std::vector<vk::Image>& GetImages() const { return images_; }
  const std::vector<vk::raii::ImageView>& GetImageViews() const { return image_views_; }
  vk::Format GetImageFormat() const { return format_.format; }
  vk::Extent2D GetExtent() const { return extent_; }
  std::uint32_t GetImageCount() const { return static_cast<std::uint32_t>(images_.size()); }

  vk::Image GetDepthImage() const { return *depth_image_; }
  const vk::raii::ImageView& GetDepthView() const { return depth_view_; }
  vk::Format GetDepthFormat() const { return depth_format_; }

private:
  void Create();
  void CreateImageViews();
  void CreateDepthResources();
  vk::Format FindDepthFormat() const;
  std::uint32_t FindMemoryType(std::uint32_t type_filter, vk::MemoryPropertyFlags properties) const;
  vk::SurfaceFormatKHR ChooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats) const;
  vk::PresentModeKHR ChoosePresentMode(const std::vector<vk::PresentModeKHR>& modes) const;
  vk::Extent2D ChooseExtent(const vk::SurfaceCapabilitiesKHR& caps) const;
  std::uint32_t ChooseImageCount(const vk::SurfaceCapabilitiesKHR& caps) const;

  Window& window_;
  Device& device_;

  vk::raii::SwapchainKHR swapchain_ = nullptr;
  std::vector<vk::Image> images_;
  std::vector<vk::raii::ImageView> image_views_;
  vk::SurfaceFormatKHR format_{};
  vk::Extent2D extent_{};

  vk::Format depth_format_ = vk::Format::eUndefined;
  vk::raii::Image depth_image_ = nullptr;
  vk::raii::DeviceMemory depth_memory_ = nullptr;
  vk::raii::ImageView depth_view_ = nullptr;
};

// ---------------------------------------------------------------------------: Implementation

Swapchain::Swapchain(Window& window, Device& device) : window_(window), device_(device) {
  Create();
  CreateImageViews();
  CreateDepthResources();
}

void Swapchain::Recreate() {
  // If minimized, wait until the window has a non-zero size again.
  auto extent = window_.GetFramebufferSize();
  while (extent.width == 0 || extent.height == 0) {
    window_.WaitEvents();
    extent = window_.GetFramebufferSize();
  }

  device_.GetLogicalDevice().waitIdle();

  depth_view_ = nullptr;
  depth_image_ = nullptr;
  depth_memory_ = nullptr;
  image_views_.clear();
  swapchain_ = nullptr;

  Create();
  CreateImageViews();
  CreateDepthResources();
}

void Swapchain::Create() {
  const auto& physical = device_.GetPhysicalDevice();
  const auto& surface = device_.GetSurface();

  auto caps = physical.getSurfaceCapabilitiesKHR(*surface);
  auto formats = physical.getSurfaceFormatsKHR(*surface);
  auto modes = physical.getSurfacePresentModesKHR(*surface);

  format_ = ChooseSurfaceFormat(formats);
  auto present_mode = ChoosePresentMode(modes);
  extent_ = ChooseExtent(caps);
  auto image_count = ChooseImageCount(caps);

  vk::SwapchainCreateInfoKHR create_info{
      .surface = *surface,
      .minImageCount = image_count,
      .imageFormat = format_.format,
      .imageColorSpace = format_.colorSpace,
      .imageExtent = extent_,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .preTransform = caps.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = present_mode,
      .clipped = vk::True,
  };

  swapchain_ = vk::raii::SwapchainKHR(device_.GetLogicalDevice(), create_info);
  images_ = swapchain_.getImages();
}

void Swapchain::CreateImageViews() {
  image_views_.reserve(images_.size());
  for (auto image : images_) {
    vk::ImageViewCreateInfo view_info{
        .image = image,
        .viewType = vk::ImageViewType::e2D,
        .format = format_.format,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };
    image_views_.emplace_back(device_.GetLogicalDevice(), view_info);
  }
}

vk::SurfaceFormatKHR Swapchain::ChooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats) const {
  auto it = std::ranges::find_if(formats, [](const auto& f) {
    return f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
  });
  return it != formats.end() ? *it : formats.front();
}

vk::PresentModeKHR Swapchain::ChoosePresentMode(const std::vector<vk::PresentModeKHR>& modes) const {
  bool has_mailbox = std::ranges::any_of(modes, [](auto m) { return m == vk::PresentModeKHR::eMailbox; });
  return has_mailbox ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
}

vk::Extent2D Swapchain::ChooseExtent(const vk::SurfaceCapabilitiesKHR& caps) const {
  if (caps.currentExtent.width != std::numeric_limits<std::uint32_t>::max()) {
    return caps.currentExtent;
  }
  auto fb = window_.GetFramebufferSize();
  return vk::Extent2D{
      std::clamp(fb.width, caps.minImageExtent.width, caps.maxImageExtent.width),
      std::clamp(fb.height, caps.minImageExtent.height, caps.maxImageExtent.height),
  };
}

std::uint32_t Swapchain::ChooseImageCount(const vk::SurfaceCapabilitiesKHR& caps) const {
  std::uint32_t count = std::max(3u, caps.minImageCount);
  if (caps.maxImageCount > 0 && count > caps.maxImageCount) {
    count = caps.maxImageCount;
  }
  return count;
}

void Swapchain::CreateDepthResources() {
  depth_format_ = FindDepthFormat();
  const auto& dev = device_.GetLogicalDevice();

  vk::ImageCreateInfo image_info{
      .imageType = vk::ImageType::e2D,
      .format = depth_format_,
      .extent = {extent_.width, extent_.height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = vk::SampleCountFlagBits::e1,
      .tiling = vk::ImageTiling::eOptimal,
      .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
      .sharingMode = vk::SharingMode::eExclusive,
      .initialLayout = vk::ImageLayout::eUndefined,
  };
  depth_image_ = vk::raii::Image(dev, image_info);

  vk::MemoryRequirements reqs = depth_image_.getMemoryRequirements();
  vk::MemoryAllocateInfo alloc_info{
      .allocationSize = reqs.size,
      .memoryTypeIndex = FindMemoryType(reqs.memoryTypeBits,
                                        vk::MemoryPropertyFlagBits::eDeviceLocal),
  };
  depth_memory_ = vk::raii::DeviceMemory(dev, alloc_info);
  depth_image_.bindMemory(*depth_memory_, 0);

  vk::ImageViewCreateInfo view_info{
      .image = *depth_image_,
      .viewType = vk::ImageViewType::e2D,
      .format = depth_format_,
      .subresourceRange = {
          .aspectMask = vk::ImageAspectFlagBits::eDepth,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
      },
  };
  depth_view_ = vk::raii::ImageView(dev, view_info);
}

vk::Format Swapchain::FindDepthFormat() const {
  // Prefer a pure-depth format; fall back to depth-stencil combos if the GPU
  // doesn't advertise the first choice.
  const std::array candidates = {
      vk::Format::eD32Sfloat,
      vk::Format::eD32SfloatS8Uint,
      vk::Format::eD24UnormS8Uint,
  };
  for (auto fmt : candidates) {
    auto props = device_.GetPhysicalDevice().getFormatProperties(fmt);
    if ((props.optimalTilingFeatures & vk::FormatFeatureFlagBits::eDepthStencilAttachment)
        == vk::FormatFeatureFlagBits::eDepthStencilAttachment) {
      return fmt;
    }
  }
  throw std::runtime_error("Failed to find supported depth format");
}

std::uint32_t Swapchain::FindMemoryType(std::uint32_t type_filter,
                                        vk::MemoryPropertyFlags properties) const {
  auto props = device_.GetPhysicalDevice().getMemoryProperties();
  for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    bool type_ok = (type_filter & (1u << i)) != 0;
    bool props_ok = (props.memoryTypes[i].propertyFlags & properties) == properties;
    if (type_ok && props_ok) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type for depth image");
}

}  // namespace engine
