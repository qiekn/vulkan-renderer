module;

#include <glm/glm.hpp>

export module engine.forward_pass;

import vulkan;
import std;
import engine.material_bindings;
import engine.model;
import engine.pipeline;
import engine.render_pass;
import engine.uniform;

namespace engine {

// ---------------------------------------------------------------------------: ForwardPass

// Walks the Model's scene graph once per instance, pushing a per-node world
// matrix + per-material push constant, binding the material's descriptor set
// (set=1), then issuing an indexed draw per primitive. Instance world matrices
// come in via SetInstances() so the app can swap them frame-to-frame without
// touching the pass internals.
export class ForwardPass : public RenderPass {
 public:
  ForwardPass(Pipeline& pipeline, UniformBufferSet& ubo_set, Model& model,
              MaterialBindings& materials)
      : RenderPass("Forward"),
        pipeline_(pipeline),
        ubo_set_(ubo_set),
        model_(model),
        materials_(materials) {}

  void SetInstances(std::span<const glm::mat4> instances) { instances_ = instances; }

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

    if (instances_.empty()) {
      return;
    }
    for (const glm::mat4& instance : instances_) {
      for (const Node* root : model_.root_nodes) {
        DrawNode(ctx.cmd, root, instance);
      }
    }
  }

  void EndPass(RenderContext& ctx) override {
    ctx.cmd.endRendering();
  }

 private:
  void DrawNode(const vk::raii::CommandBuffer& cmd, const Node* node,
                const glm::mat4& parent_matrix) {
    const glm::mat4 world = parent_matrix * node->GetLocalMatrix();

    for (const MeshPrimitive& prim : node->primitives) {
      DrawPushConstants push{};
      push.model = world;

      if (prim.material_index >= 0 &&
          prim.material_index < static_cast<int>(model_.materials.size())) {
        const Material& mat = model_.materials[prim.material_index];
        push.material.base_color_factor = mat.base_color_factor;
        push.material.emissive_factor = glm::vec4(mat.emissive_factor, 0.0f);
        push.material.metallic_factor = mat.metallic_factor;
        push.material.roughness_factor = mat.roughness_factor;
        push.material.base_color_tex_set = mat.base_color_texture >= 0 ? 1 : -1;
        push.material.metallic_roughness_tex_set = mat.metallic_roughness_texture >= 0 ? 1 : -1;
        push.material.normal_tex_set = mat.normal_texture >= 0 ? 1 : -1;
        push.material.occlusion_tex_set = mat.occlusion_texture >= 0 ? 1 : -1;
        push.material.emissive_tex_set = mat.emissive_texture >= 0 ? 1 : -1;
      } else {
        // Neutral fallback: white-ish dielectric, moderately rough, no emissive.
        push.material.base_color_factor = glm::vec4(1.0f);
        push.material.emissive_factor = glm::vec4(0.0f);
        push.material.metallic_factor = 0.0f;
        push.material.roughness_factor = 0.8f;
        push.material.base_color_tex_set = -1;
        push.material.metallic_roughness_tex_set = -1;
        push.material.normal_tex_set = -1;
        push.material.occlusion_tex_set = -1;
        push.material.emissive_tex_set = -1;
      }

      vk::DescriptorSet material_set = materials_.GetSet(prim.material_index);
      cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline_.GetLayout(),
                             1, material_set, nullptr);

      cmd.pushConstants<DrawPushConstants>(
          *pipeline_.GetLayout(),
          vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
          0, push);

      cmd.bindVertexBuffers(0, {*prim.vertex_buffer}, {0});
      cmd.bindIndexBuffer(*prim.index_buffer, 0, vk::IndexType::eUint32);
      cmd.drawIndexed(prim.index_count, 1, 0, 0, 0);
    }

    for (const Node* child : node->children) {
      DrawNode(cmd, child, world);
    }
  }

  Pipeline& pipeline_;
  UniformBufferSet& ubo_set_;
  Model& model_;
  MaterialBindings& materials_;
  std::span<const glm::mat4> instances_;
};

}  // namespace engine
