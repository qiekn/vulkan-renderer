export module engine.pipeline;

import vulkan;
import std;
import engine.device;

namespace engine {

// ---------------------------------------------------------------------------: Pipeline

// Minimal graphics pipeline for dynamic rendering: loads one SPIR-V blob that
// contains both vertMain and fragMain entry points (produced by slangc) and
// builds a pipeline with a single color attachment and no vertex input.
export class Pipeline {
 public:
  Pipeline(Device& device, const std::filesystem::path& spirv_path, vk::Format color_format);

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;
  Pipeline(Pipeline&&) = delete;
  Pipeline& operator=(Pipeline&&) = delete;

  const vk::raii::Pipeline& GetHandle() const { return pipeline_; }
  const vk::raii::PipelineLayout& GetLayout() const { return layout_; }

 private:
  static std::vector<char> ReadFile(const std::filesystem::path& path);
  vk::raii::ShaderModule CreateShaderModule(const std::vector<char>& code) const;

  Device& device_;
  vk::raii::PipelineLayout layout_ = nullptr;
  vk::raii::Pipeline pipeline_ = nullptr;
};

// ---------------------------------------------------------------------------: Implementation

Pipeline::Pipeline(Device& device, const std::filesystem::path& spirv_path, vk::Format color_format)
    : device_(device) {
  auto code = ReadFile(spirv_path);
  auto shader_module = CreateShaderModule(code);

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

  // No descriptors, no push constants yet.
  vk::PipelineLayoutCreateInfo layout_info{};
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

std::vector<char> Pipeline::ReadFile(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open shader file: " + path.string());
  }
  std::vector<char> buffer(static_cast<std::size_t>(file.tellg()));
  file.seekg(0);
  file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  return buffer;
}

vk::raii::ShaderModule Pipeline::CreateShaderModule(const std::vector<char>& code) const {
  vk::ShaderModuleCreateInfo create_info{
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const std::uint32_t*>(code.data()),
  };
  return vk::raii::ShaderModule(device_.GetLogicalDevice(), create_info);
}

}  // namespace engine
