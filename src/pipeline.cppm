export module engine.pipeline;

import vulkan;
import std;
import engine.device;

namespace engine {

// ---------------------------------------------------------------------------: PipelineConfig

// All the knobs the forward pipeline actually needs. Keeping this as a plain
// struct (rather than a builder) keeps construction boring: callers fill it out
// and pass it in. New passes add fields here instead of growing the Pipeline
// constructor's positional arg list.
export struct PipelineConfig {
  vk::ShaderModule shader_module = nullptr;
  vk::Format color_format = vk::Format::eUndefined;
  vk::Format depth_format = vk::Format::eUndefined;  // eUndefined = no depth attachment

  vk::VertexInputBindingDescription vertex_binding{};
  std::span<const vk::VertexInputAttributeDescription> vertex_attributes{};

  // Set layouts bound in order (set=0, set=1, ...). An empty span produces a
  // pipeline layout with no descriptor sets at all.
  std::span<const vk::DescriptorSetLayout> descriptor_set_layouts{};
  std::span<const vk::PushConstantRange> push_constant_ranges{};

  vk::CullModeFlags cull_mode = vk::CullModeFlagBits::eBack;
  vk::FrontFace front_face = vk::FrontFace::eCounterClockwise;

  bool depth_test = true;
  bool depth_write = true;
};

// ---------------------------------------------------------------------------: Pipeline

// Graphics pipeline for dynamic rendering. Shader module, formats, vertex input,
// descriptor layout, push constants, cull mode, and depth state are all fed
// through PipelineConfig.
export class Pipeline {
public:
  Pipeline(Device& device, const PipelineConfig& config);

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;

  const vk::raii::Pipeline& GetHandle() const { return pipeline_; }
  const vk::raii::PipelineLayout& GetLayout() const { return layout_; }

private:
  Device& device_;
  vk::raii::PipelineLayout layout_ = nullptr;
  vk::raii::Pipeline pipeline_ = nullptr;
};

// ---------------------------------------------------------------------------: Implementation

Pipeline::Pipeline(Device& device, const PipelineConfig& config) : device_(device) {
  std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
      vk::PipelineShaderStageCreateInfo{
          .stage = vk::ShaderStageFlagBits::eVertex,
          .module = config.shader_module,
          .pName = "vertMain",
      },
      vk::PipelineShaderStageCreateInfo{
          .stage = vk::ShaderStageFlagBits::eFragment,
          .module = config.shader_module,
          .pName = "fragMain",
      },
  };

  // stride==0 means "no vertex buffer bound"; we detect that here so the old
  // hardcoded-triangle path still works without touching the shader.
  const bool has_vertex_input = config.vertex_binding.stride > 0;
  vk::PipelineVertexInputStateCreateInfo vertex_input{};
  if (has_vertex_input) {
    vertex_input.vertexBindingDescriptionCount = 1;
    vertex_input.pVertexBindingDescriptions = &config.vertex_binding;
    vertex_input.vertexAttributeDescriptionCount =
        static_cast<std::uint32_t>(config.vertex_attributes.size());
    vertex_input.pVertexAttributeDescriptions = config.vertex_attributes.data();
  }

  vk::PipelineInputAssemblyStateCreateInfo input_assembly{
      .topology = vk::PrimitiveTopology::eTriangleList,
  };

  vk::PipelineViewportStateCreateInfo viewport_state{
      .viewportCount = 1,
      .scissorCount = 1,
  };

  vk::PipelineRasterizationStateCreateInfo rasterizer{
      .depthClampEnable = vk::False,
      .rasterizerDiscardEnable = vk::False,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = config.cull_mode,
      .frontFace = config.front_face,
      .depthBiasEnable = vk::False,
      .lineWidth = 1.0f,
  };

  vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = vk::False,
  };

  const bool has_depth = config.depth_format != vk::Format::eUndefined;
  vk::PipelineDepthStencilStateCreateInfo depth_stencil{
      .depthTestEnable = has_depth && config.depth_test ? vk::True : vk::False,
      .depthWriteEnable = has_depth && config.depth_write ? vk::True : vk::False,
      .depthCompareOp = vk::CompareOp::eLess,
      .depthBoundsTestEnable = vk::False,
      .stencilTestEnable = vk::False,
  };

  vk::PipelineColorBlendAttachmentState color_blend_attachment{
      .blendEnable = vk::False,
      .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG
                      | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };

  vk::PipelineColorBlendStateCreateInfo color_blending{
      .logicOpEnable = vk::False,
      .attachmentCount = 1,
      .pAttachments = &color_blend_attachment,
  };

  std::array dynamic_states = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
  vk::PipelineDynamicStateCreateInfo dynamic_state{
      .dynamicStateCount = static_cast<std::uint32_t>(dynamic_states.size()),
      .pDynamicStates = dynamic_states.data(),
  };

  vk::PipelineLayoutCreateInfo layout_info{};
  if (!config.descriptor_set_layouts.empty()) {
    layout_info.setLayoutCount = static_cast<std::uint32_t>(config.descriptor_set_layouts.size());
    layout_info.pSetLayouts = config.descriptor_set_layouts.data();
  }
  if (!config.push_constant_ranges.empty()) {
    layout_info.pushConstantRangeCount =
        static_cast<std::uint32_t>(config.push_constant_ranges.size());
    layout_info.pPushConstantRanges = config.push_constant_ranges.data();
  }
  layout_ = vk::raii::PipelineLayout(device_.GetLogicalDevice(), layout_info);

  vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> chain{
      vk::GraphicsPipelineCreateInfo{
          .stageCount = static_cast<std::uint32_t>(stages.size()),
          .pStages = stages.data(),
          .pVertexInputState = &vertex_input,
          .pInputAssemblyState = &input_assembly,
          .pViewportState = &viewport_state,
          .pRasterizationState = &rasterizer,
          .pMultisampleState = &multisampling,
          .pDepthStencilState = has_depth ? &depth_stencil : nullptr,
          .pColorBlendState = &color_blending,
          .pDynamicState = &dynamic_state,
          .layout = layout_,
          .renderPass = nullptr,
      },
      vk::PipelineRenderingCreateInfo{
          .colorAttachmentCount = 1,
          .pColorAttachmentFormats = &config.color_format,
          .depthAttachmentFormat = config.depth_format,
      },
  };

  pipeline_ = vk::raii::Pipeline(device_.GetLogicalDevice(), nullptr,
                                 chain.get<vk::GraphicsPipelineCreateInfo>());
}

}  // namespace engine
