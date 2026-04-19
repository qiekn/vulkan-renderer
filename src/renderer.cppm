export module engine.renderer;

import vulkan;
import std;
import engine.window;
import engine.device;
import engine.swapchain;
import engine.render_pass;

namespace engine {

// ---------------------------------------------------------------------------: Renderer

// Owns per-frame command buffers + sync objects and drives a single DrawFrame
// call. Layout transitions around the swapchain image happen here; everything
// between beginRendering and endRendering is delegated to RenderPassManager.
export class Renderer {
 public:
  static constexpr std::uint32_t kMaxFramesInFlight = 2;

  Renderer(Window& window, Device& device, Swapchain& swapchain, RenderPassManager& passes);

  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  Renderer(Renderer&&) = delete;
  Renderer& operator=(Renderer&&) = delete;

  void DrawFrame();
  void WaitIdle() { device_.GetLogicalDevice().waitIdle(); }

  std::uint32_t GetCurrentFrame() const { return frame_index_; }

 private:
  void CreateCommandPool();
  void CreateCommandBuffers();
  void CreateSyncObjects();
  void RecordCommandBuffer(vk::raii::CommandBuffer& cmd, std::uint32_t image_index);
  void TransitionImageLayout(vk::raii::CommandBuffer& cmd,
                             vk::Image image,
                             vk::ImageLayout old_layout,
                             vk::ImageLayout new_layout,
                             vk::AccessFlags2 src_access,
                             vk::AccessFlags2 dst_access,
                             vk::PipelineStageFlags2 src_stage,
                             vk::PipelineStageFlags2 dst_stage);

  Window& window_;
  Device& device_;
  Swapchain& swapchain_;
  RenderPassManager& passes_;

  vk::raii::CommandPool command_pool_ = nullptr;
  std::vector<vk::raii::CommandBuffer> command_buffers_;
  std::vector<vk::raii::Semaphore> image_available_semaphores_;
  std::vector<vk::raii::Semaphore> render_finished_semaphores_;
  std::vector<vk::raii::Fence> in_flight_fences_;
  std::uint32_t frame_index_ = 0;
};

// ---------------------------------------------------------------------------: Implementation

Renderer::Renderer(Window& window, Device& device, Swapchain& swapchain, RenderPassManager& passes)
    : window_(window), device_(device), swapchain_(swapchain), passes_(passes) {
  CreateCommandPool();
  CreateCommandBuffers();
  CreateSyncObjects();
}

void Renderer::CreateCommandPool() {
  vk::CommandPoolCreateInfo info{
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = device_.GetGraphicsQueueFamily(),
  };
  command_pool_ = vk::raii::CommandPool(device_.GetLogicalDevice(), info);
}

void Renderer::CreateCommandBuffers() {
  vk::CommandBufferAllocateInfo info{
      .commandPool = command_pool_,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = kMaxFramesInFlight,
  };
  command_buffers_ = vk::raii::CommandBuffers(device_.GetLogicalDevice(), info);
}

void Renderer::CreateSyncObjects() {
  for (std::uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    image_available_semaphores_.emplace_back(device_.GetLogicalDevice(), vk::SemaphoreCreateInfo{});
    in_flight_fences_.emplace_back(device_.GetLogicalDevice(),
                                   vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
  }
  for (std::uint32_t i = 0; i < swapchain_.GetImageCount(); ++i) {
    render_finished_semaphores_.emplace_back(device_.GetLogicalDevice(), vk::SemaphoreCreateInfo{});
  }
}

void Renderer::DrawFrame() {
  const auto& dev = device_.GetLogicalDevice();

  auto wait_result = dev.waitForFences(*in_flight_fences_[frame_index_], vk::True,
                                       std::numeric_limits<std::uint64_t>::max());
  if (wait_result != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to wait for in-flight fence");
  }

  std::uint32_t image_index = 0;
  try {
    auto [result, index] = swapchain_.GetHandle().acquireNextImage(
        std::numeric_limits<std::uint64_t>::max(), *image_available_semaphores_[frame_index_], nullptr);
    if (result == vk::Result::eSuboptimalKHR) {
      swapchain_.Recreate();
      return;
    }
    if (result != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to acquire swapchain image");
    }
    image_index = index;
  } catch (const vk::OutOfDateKHRError&) {
    swapchain_.Recreate();
    return;
  }

  dev.resetFences(*in_flight_fences_[frame_index_]);

  auto& cmd = command_buffers_[frame_index_];
  cmd.reset();
  RecordCommandBuffer(cmd, image_index);

  vk::PipelineStageFlags wait_stage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
  vk::SubmitInfo submit_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*image_available_semaphores_[frame_index_],
      .pWaitDstStageMask = &wait_stage,
      .commandBufferCount = 1,
      .pCommandBuffers = &*cmd,
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &*render_finished_semaphores_[image_index],
  };
  device_.GetGraphicsQueue().submit(submit_info, *in_flight_fences_[frame_index_]);

  vk::PresentInfoKHR present_info{
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*render_finished_semaphores_[image_index],
      .swapchainCount = 1,
      .pSwapchains = &*swapchain_.GetHandle(),
      .pImageIndices = &image_index,
  };
  try {
    auto result = device_.GetGraphicsQueue().presentKHR(present_info);
    if (result == vk::Result::eSuboptimalKHR || window_.WasResized()) {
      window_.ClearResized();
      swapchain_.Recreate();
    } else if (result != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to present swapchain image");
    }
  } catch (const vk::OutOfDateKHRError&) {
    window_.ClearResized();
    swapchain_.Recreate();
  }

  frame_index_ = (frame_index_ + 1) % kMaxFramesInFlight;
}

void Renderer::RecordCommandBuffer(vk::raii::CommandBuffer& cmd, std::uint32_t image_index) {
  cmd.begin(vk::CommandBufferBeginInfo{});

  TransitionImageLayout(cmd, swapchain_.GetImages()[image_index],
                        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                        vk::AccessFlags2{}, vk::AccessFlagBits2::eColorAttachmentWrite,
                        vk::PipelineStageFlagBits2::eTopOfPipe,
                        vk::PipelineStageFlagBits2::eColorAttachmentOutput);

  RenderContext ctx{
      .cmd = cmd,
      .target_view = *swapchain_.GetImageViews()[image_index],
      .target_extent = swapchain_.GetExtent(),
      .frame_index = frame_index_,
  };
  passes_.Execute(ctx);

  TransitionImageLayout(cmd, swapchain_.GetImages()[image_index],
                        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR,
                        vk::AccessFlagBits2::eColorAttachmentWrite, vk::AccessFlags2{},
                        vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                        vk::PipelineStageFlagBits2::eBottomOfPipe);

  cmd.end();
}

void Renderer::TransitionImageLayout(vk::raii::CommandBuffer& cmd,
                                     vk::Image image,
                                     vk::ImageLayout old_layout,
                                     vk::ImageLayout new_layout,
                                     vk::AccessFlags2 src_access,
                                     vk::AccessFlags2 dst_access,
                                     vk::PipelineStageFlags2 src_stage,
                                     vk::PipelineStageFlags2 dst_stage) {
  vk::ImageMemoryBarrier2 barrier{
      .srcStageMask = src_stage,
      .srcAccessMask = src_access,
      .dstStageMask = dst_stage,
      .dstAccessMask = dst_access,
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = image,
      .subresourceRange = {
          .aspectMask = vk::ImageAspectFlagBits::eColor,
          .baseMipLevel = 0,
          .levelCount = 1,
          .baseArrayLayer = 0,
          .layerCount = 1,
      },
  };
  cmd.pipelineBarrier2(vk::DependencyInfo{
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
  });
}

}  // namespace engine
