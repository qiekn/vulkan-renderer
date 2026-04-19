module;

#include <glm/glm.hpp>

export module engine.uniform;

import vulkan;
import std;
import engine.device;

namespace engine {

// ---------------------------------------------------------------------------: UBO layout

// Matches the `ubo` ConstantBuffer in triangle.slang. GLM produces column-major
// matrices, which is what Slang's SPIR-V backend expects when compiled with
// `-matrix-layout-column-major`; std140 layout is satisfied automatically
// because every member is a 4x4 matrix aligned to 16 bytes.
export struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
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
      .stageFlags = vk::ShaderStageFlagBits::eVertex,
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
