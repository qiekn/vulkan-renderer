export module engine.forward_pass;

import vulkan;
import std;
import engine.pipeline;
import engine.render_pass;
import engine.uniform;

namespace engine {

// ---------------------------------------------------------------------------: ForwardPass

// Single-attachment forward pass that clears the color target and renders a
// hardcoded triangle via the supplied pipeline. Binds the per-frame MVP UBO
// descriptor set before issuing the draw.
export class ForwardPass : public RenderPass {
 public:
  ForwardPass(Pipeline& pipeline, UniformBufferSet& ubo_set)
      : RenderPass("Forward"), pipeline_(pipeline), ubo_set_(ubo_set) {}

 protected:
  void BeginPass(RenderContext& ctx) override {
    vk::ClearValue clear_color = vk::ClearColorValue(std::array<float, 4>{0.01f, 0.01f, 0.03f, 1.0f});
    vk::RenderingAttachmentInfo color_attachment{
        .imageView = ctx.target_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eStore,
        .clearValue = clear_color,
    };
    vk::RenderingInfo rendering_info{
        .renderArea = vk::Rect2D{{0, 0}, ctx.target_extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
    };
    ctx.cmd.beginRendering(rendering_info);
  }

  void Render(RenderContext& ctx) override {
    ctx.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_.GetHandle());
    ctx.cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_.GetLayout(),
                               0, *ubo_set_.GetSet(ctx.frame_index), nullptr);

    vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = static_cast<float>(ctx.target_extent.width),
        .height = static_cast<float>(ctx.target_extent.height),
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    ctx.cmd.setViewport(0, viewport);
    ctx.cmd.setScissor(0, vk::Rect2D{{0, 0}, ctx.target_extent});

    ctx.cmd.draw(3, 1, 0, 0);
  }

  void EndPass(RenderContext& ctx) override {
    ctx.cmd.endRendering();
  }

 private:
  Pipeline& pipeline_;
  UniformBufferSet& ubo_set_;
};

}  // namespace engine
