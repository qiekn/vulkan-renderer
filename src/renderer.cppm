export module engine.renderer;

import vulkan;
import std;
import engine.window;
import engine.device;
import engine.swapchain;
import engine.pipeline;

namespace engine {

// ---------------------------------------------------------------------------: Renderer

// Owns per-frame command buffers + sync objects and drives a single DrawFrame
// call. Uses dynamic rendering directly against the swapchain image.
export class Renderer {
 public:
  static constexpr std::uint32_t kMaxFramesInFlight = 2;

  Renderer(Window& window, Device& device, Swapchain& swapchain, Pipeline& pipeline);

  Renderer(const Renderer&) = delete;
  Renderer& operator=(const Renderer&) = delete;
  Renderer(Renderer&&) = delete;
  Renderer& operator=(Renderer&&) = delete;

  void DrawFrame();
  void WaitIdle() { device_.GetLogicalDevice().waitIdle(); }

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
  Pipeline& pipeline_;

  vk::raii::CommandPool command_pool_ = nullptr;
  std::vector<vk::raii::CommandBuffer> command_buffers_;
  std::vector<vk::raii::Semaphore> image_available_semaphores_;
  std::vector<vk::raii::Semaphore> render_finished_semaphores_;
  std::vector<vk::raii::Fence> in_flight_fences_;
  std::uint32_t frame_index_ = 0;
};

// ---------------------------------------------------------------------------: Implementation

Renderer::Renderer(Window& window, Device& device, Swapchain& swapchain, Pipeline& pipeline)
    : window_(window), device_(device), swapchain_(swapchain), pipeline_(pipeline) {
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
  // image_available / in_flight are per-in-flight-frame; render_finished is
  // per-swapchain-image so that present never waits on a semaphore that
  // belongs to a different image.
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

  // 1. Wait for this frame slot to be free.
  auto wait_result = dev.waitForFences(*in_flight_fences_[frame_index_], vk::True,
                                       std::numeric_limits<std::uint64_t>::max());
  if (wait_result != vk::Result::eSuccess) {
    throw std::runtime_error("Failed to wait for in-flight fence");
  }

  // 2. Acquire next swapchain image.
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

  // 3. Record commands for this frame.
  auto& cmd = command_buffers_[frame_index_];
  cmd.reset();
  RecordCommandBuffer(cmd, image_index);

  // 4. Submit.
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

  // 5. Present.
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

  // Undefined → ColorAttachmentOptimal before rendering.
  TransitionImageLayout(cmd, swapchain_.GetImages()[image_index],
                        vk::ImageLayout::eUndefined, vk::ImageLayout::eColorAttachmentOptimal,
                        vk::AccessFlags2{}, vk::AccessFlagBits2::eColorAttachmentWrite,
                        vk::PipelineStageFlagBits2::eTopOfPipe,
                        vk::PipelineStageFlagBits2::eColorAttachmentOutput);

  vk::ClearValue clear_color = vk::ClearColorValue(std::array<float, 4>{0.01f, 0.01f, 0.03f, 1.0f});
  vk::RenderingAttachmentInfo color_attachment{
      .imageView = *swapchain_.GetImageViews()[image_index],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = clear_color,
  };

  vk::RenderingInfo rendering_info{
      .renderArea = vk::Rect2D{{0, 0}, swapchain_.GetExtent()},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_attachment,
  };

  cmd.beginRendering(rendering_info);
  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_.GetHandle());

  auto extent = swapchain_.GetExtent();
  vk::Viewport viewport{
      .x = 0.0f,
      .y = 0.0f,
      .width = static_cast<float>(extent.width),
      .height = static_cast<float>(extent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
  };
  cmd.setViewport(0, viewport);
  cmd.setScissor(0, vk::Rect2D{{0, 0}, extent});

  cmd.draw(3, 1, 0, 0);
  cmd.endRendering();

  // ColorAttachmentOptimal → PresentSrcKHR before presenting.
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
