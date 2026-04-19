module;

#include <glm/glm.hpp>

export module engine.uniform;

import vulkan;
import std;
import engine.device;

namespace engine {

// ---------------------------------------------------------------------------: UBO layout

// Matches the `ubo` ConstantBuffer in pbr.slang. All members are vec4 / mat4 so
// std140 / std430 layout rules line up 1:1 with the GLM memory layout (column-
// major, 16-byte-aligned). Keep scalar floats together at the tail and pad them
// to a multiple of 16 bytes so the struct stride is predictable.
export struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
  glm::vec4 light_positions[4];
  glm::vec4 light_colors[4];
  glm::vec4 cam_pos;
  float exposure;
  float gamma;
  float _pad0 = 0.0f;
  float _pad1 = 0.0f;
};

// Material knobs shipped as push constants per draw. The BRDF in pbr.slang reads
// base color, metallic, and roughness straight out of this block.
export struct MaterialPushConstants {
  glm::vec4 base_color;
  float metallic;
  float roughness;
  float _pad0 = 0.0f;
  float _pad1 = 0.0f;
};

// ---------------------------------------------------------------------------: UniformBufferSet

// Per-frame-in-flight UBO ring: one host-visible+coherent buffer per frame,
// persistently mapped, backed by a single descriptor set each. Update(frame, ubo)
// writes the matrices via the mapped pointer; GetSet(frame) is bound by the
// render pass for the corresponding command buffer.
export class UniformBufferSet {
public:
  UniformBufferSet(Device& device, std::uint32_t frame_count);

  UniformBufferSet(const UniformBufferSet&) = delete;
  UniformBufferSet& operator=(const UniformBufferSet&) = delete;
  UniformBufferSet(UniformBufferSet&&) = delete;
  UniformBufferSet& operator=(UniformBufferSet&&) = delete;

  const vk::raii::DescriptorSetLayout& GetLayout() const { return layout_; }
  const vk::raii::DescriptorSet& GetSet(std::uint32_t frame) const { return sets_[frame]; }

  void Update(std::uint32_t frame, const UniformBufferObject& ubo);

private:
  std::uint32_t FindMemoryType(std::uint32_t type_filter, vk::MemoryPropertyFlags properties) const;

  struct Frame {
    vk::raii::Buffer buffer = nullptr;
    vk::raii::DeviceMemory memory = nullptr;
    void* mapped = nullptr;
  };

  Device& device_;
  vk::raii::DescriptorSetLayout layout_ = nullptr;
  vk::raii::DescriptorPool pool_ = nullptr;
  std::vector<Frame> frames_;
  std::vector<vk::raii::DescriptorSet> sets_;
};

// ---------------------------------------------------------------------------: Implementation

UniformBufferSet::UniformBufferSet(Device& device, std::uint32_t frame_count) : device_(device) {
  const auto& dev = device_.GetLogicalDevice();

  vk::DescriptorSetLayoutBinding ubo_binding{
      .binding = 0,
      .descriptorType = vk::DescriptorType::eUniformBuffer,
      .descriptorCount = 1,
      // Matrices feed the vertex stage; lights + cam_pos feed the fragment stage.
      .stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
  };
  vk::DescriptorSetLayoutCreateInfo layout_info{
      .bindingCount = 1,
      .pBindings = &ubo_binding,
  };
  layout_ = vk::raii::DescriptorSetLayout(dev, layout_info);

  vk::DescriptorPoolSize pool_size{
      .type = vk::DescriptorType::eUniformBuffer,
      .descriptorCount = frame_count,
  };
  vk::DescriptorPoolCreateInfo pool_info{
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = frame_count,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
  };
  pool_ = vk::raii::DescriptorPool(dev, pool_info);

  frames_.reserve(frame_count);
  const vk::DeviceSize buffer_size = sizeof(UniformBufferObject);
  for (std::uint32_t i = 0; i < frame_count; ++i) {
    Frame frame;
    vk::BufferCreateInfo buffer_info{
        .size = buffer_size,
        .usage = vk::BufferUsageFlagBits::eUniformBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    frame.buffer = vk::raii::Buffer(dev, buffer_info);

    vk::MemoryRequirements reqs = frame.buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo alloc_info{
        .allocationSize = reqs.size,
        .memoryTypeIndex = FindMemoryType(
            reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    };
    frame.memory = vk::raii::DeviceMemory(dev, alloc_info);
    frame.buffer.bindMemory(*frame.memory, 0);
    frame.mapped = frame.memory.mapMemory(0, buffer_size);
    frames_.push_back(std::move(frame));
  }

  std::vector<vk::DescriptorSetLayout> layouts(frame_count, *layout_);
  vk::DescriptorSetAllocateInfo set_alloc_info{
      .descriptorPool = *pool_,
      .descriptorSetCount = frame_count,
      .pSetLayouts = layouts.data(),
  };
  sets_ = vk::raii::DescriptorSets(dev, set_alloc_info);

  for (std::uint32_t i = 0; i < frame_count; ++i) {
    vk::DescriptorBufferInfo buffer_info{
        .buffer = *frames_[i].buffer,
        .offset = 0,
        .range = sizeof(UniformBufferObject),
    };
    vk::WriteDescriptorSet write{
        .dstSet = *sets_[i],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .pBufferInfo = &buffer_info,
    };
    dev.updateDescriptorSets(write, nullptr);
  }
}

void UniformBufferSet::Update(std::uint32_t frame, const UniformBufferObject& ubo) {
  std::memcpy(frames_[frame].mapped, &ubo, sizeof(ubo));
}

std::uint32_t UniformBufferSet::FindMemoryType(std::uint32_t type_filter,
                                               vk::MemoryPropertyFlags properties) const {
  auto props = device_.GetPhysicalDevice().getMemoryProperties();
  for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    bool type_ok = (type_filter & (1u << i)) != 0;
    bool props_ok = (props.memoryTypes[i].propertyFlags & properties) == properties;
    if (type_ok && props_ok) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type for uniform buffer");
}

}  // namespace engine
