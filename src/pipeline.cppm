export module engine.pipeline;

import vulkan;
import std;
import engine.device;

namespace engine {

// ---------------------------------------------------------------------------: Pipeline

// Minimal graphics pipeline for dynamic rendering. Takes an externally-owned
// shader module (containing vertMain + fragMain entry points) and builds a
// pipeline with a single color attachment and no vertex input.
export class Pipeline {
 public:
  Pipeline(Device& device, vk::ShaderModule shader_module, vk::Format color_format,
           vk::DescriptorSetLayout descriptor_set_layout = nullptr);

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

Pipeline::Pipeline(Device& device, vk::ShaderModule shader_module, vk::Format color_format,
                   vk::DescriptorSetLayout descriptor_set_layout)
    : device_(device) {
  std::array<vk::PipelineShaderStageCreateInfo, 2> stages{
      vk::PipelineShaderStageCreateInfo{
          .stage = vk::ShaderStageFlagBits::eVertex,
          .module = shader_module,
          .pName = "vertMain",
      },
      vk::PipelineShaderStageCreateInfo{
          .stage = vk::ShaderStageFlagBits::eFragment,
          .module = shader_module,
          .pName = "fragMain",
      },
  };

  // Triangle is hardcoded in the shader via SV_VertexID, no vertex input.
  vk::PipelineVertexInputStateCreateInfo vertex_input{};

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
      .cullMode = vk::CullModeFlagBits::eNone,
      .frontFace = vk::FrontFace::eCounterClockwise,
      .depthBiasEnable = vk::False,
      .lineWidth = 1.0f,
  };

  vk::PipelineMultisampleStateCreateInfo multisampling{
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = vk::False,
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

  // Descriptor set layout is passed in when the pipeline needs UBOs/samplers;
  // nullptr means no descriptors bound.
  vk::PipelineLayoutCreateInfo layout_info{};
  if (descriptor_set_layout) {
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_set_layout;
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
          .pColorBlendState = &color_blending,
          .pDynamicState = &dynamic_state,
          .layout = layout_,
          .renderPass = nullptr,
      },
      vk::PipelineRenderingCreateInfo{
          .colorAttachmentCount = 1,
          .pColorAttachmentFormats = &color_format,
      },
  };

  pipeline_ = vk::raii::Pipeline(device_.GetLogicalDevice(), nullptr,
                                 chain.get<vk::GraphicsPipelineCreateInfo>());
}

}  // namespace engine
