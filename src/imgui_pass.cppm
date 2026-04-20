module;

#include <vulkan/vulkan.h>

#include <imgui.h>
#include <imgui_impl_vulkan.h>

export module engine.imgui_pass;

import vulkan;
import std;
import engine.render_pass;

namespace engine {

// ---------------------------------------------------------------------------: ImGuiPass

// Overlays Dear ImGui draw data onto the current swapchain image. Runs after
// ForwardPass via AddDependency("Forward"), so loadOp=eLoad preserves the
// scene render; no depth attachment (ImGui is screen-space-only).
export class ImGuiPass : public RenderPass {
public:
  ImGuiPass() : RenderPass("ImGui") { AddDependency("Forward"); }

protected:
  void BeginPass(RenderContext& ctx) override {
    vk::RenderingAttachmentInfo color_attachment{
        .imageView = ctx.target_view,
        .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .loadOp = vk::AttachmentLoadOp::eLoad,
        .storeOp = vk::AttachmentStoreOp::eStore,
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
    ImDrawData* draw_data = ImGui::GetDrawData();
    if (draw_data != nullptr && draw_data->CmdListsCount > 0) {
      ImGui_ImplVulkan_RenderDrawData(draw_data, *ctx.cmd);
    }
  }

  void EndPass(RenderContext& ctx) override { ctx.cmd.endRendering(); }
};

}  // namespace engine
