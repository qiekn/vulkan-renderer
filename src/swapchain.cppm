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

private:
  void Create();
  void CreateImageViews();
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
};

// ---------------------------------------------------------------------------: Implementation

Swapchain::Swapchain(Window& window, Device& device) : window_(window), device_(device) {
  Create();
  CreateImageViews();
}

void Swapchain::Recreate() {
  // If minimized, wait until the window has a non-zero size again.
  auto extent = window_.GetFramebufferSize();
  while (extent.width == 0 || extent.height == 0) {
    window_.WaitEvents();
    extent = window_.GetFramebufferSize();
  }

  device_.GetLogicalDevice().waitIdle();

  image_views_.clear();
  swapchain_ = nullptr;

  Create();
  CreateImageViews();
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

}  // namespace engine
