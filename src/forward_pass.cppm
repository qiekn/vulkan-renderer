export module engine.forward_pass;

import vulkan;
import std;
import engine.mesh;
import engine.pipeline;
import engine.render_pass;
import engine.uniform;

namespace engine {

// ---------------------------------------------------------------------------: ForwardPass

// Single-draw forward pass: clears color + depth, binds pipeline and per-frame
// UBO, pushes the material block, binds the mesh's vertex/index buffers, and
// issues a single indexed draw. More meshes → iterate a submission list here.
export class ForwardPass : public RenderPass {
public:
  ForwardPass(Pipeline& pipeline, UniformBufferSet& ubo_set, Mesh& mesh,
              const MaterialPushConstants& material)
      : RenderPass("Forward"),
        pipeline_(pipeline),
        ubo_set_(ubo_set),
        mesh_(mesh),
        material_(material) {}

  void SetMaterial(const MaterialPushConstants& material) { material_ = material; }

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

    vk::ClearValue clear_depth = vk::ClearDepthStencilValue{.depth = 1.0f, .stencil = 0};
    vk::RenderingAttachmentInfo depth_attachment{
        .imageView = ctx.depth_view,
        .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eClear,
        .storeOp = vk::AttachmentStoreOp::eDontCare,
        .clearValue = clear_depth,
    };

    vk::RenderingInfo rendering_info{
        .renderArea = vk::Rect2D{{0, 0}, ctx.target_extent},
        .layerCount = 1,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment,
        .pDepthAttachment = ctx.depth_view ? &depth_attachment : nullptr,
    };
    ctx.cmd.beginRendering(rendering_info);
  }

  void Render(RenderContext& ctx) override {
    ctx.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline_.GetHandle());
    ctx.cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_.GetLayout(),
                               0, *ubo_set_.GetSet(ctx.frame_index), nullptr);

    ctx.cmd.pushConstants<MaterialPushConstants>(
        *pipeline_.GetLayout(),
        vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
        0, material_);

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

    ctx.cmd.bindVertexBuffers(0, {*mesh_.GetVertexBuffer()}, {0});
    ctx.cmd.bindIndexBuffer(*mesh_.GetIndexBuffer(), 0, vk::IndexType::eUint32);
    ctx.cmd.drawIndexed(mesh_.GetIndexCount(), 1, 0, 0, 0);
  }

  void EndPass(RenderContext& ctx) override {
    ctx.cmd.endRendering();
  }

private:
  Pipeline& pipeline_;
  UniformBufferSet& ubo_set_;
  Mesh& mesh_;
  MaterialPushConstants material_;
};

}  // namespace engine
