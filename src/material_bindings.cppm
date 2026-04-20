module;

#include <glm/glm.hpp>

export module engine.material_bindings;

import vulkan;
import std;
import engine.device;
import engine.model;

namespace engine {

// ---------------------------------------------------------------------------: MaterialBindings

// Owns the descriptor set layout + pool + per-material descriptor sets used to
// feed PBR textures into set=1 of the pbr.slang pipeline. A 1x1 white fallback
// fills in missing maps so every descriptor slot is always valid; the shader
// checks the push-constant `*_tex_set` flags to decide whether to actually
// multiply by the sampled value.
export class MaterialBindings {
 public:
  MaterialBindings(Device& device, Model& model);

  MaterialBindings(const MaterialBindings&) = delete;
  MaterialBindings& operator=(const MaterialBindings&) = delete;
  MaterialBindings(MaterialBindings&&) = delete;
  MaterialBindings& operator=(MaterialBindings&&) = delete;

  const vk::raii::DescriptorSetLayout& GetLayout() const { return layout_; }

  // Returns the set for `material_index` from the model, or the default set
  // when the primitive has no material (material_index == -1 or out of range).
  vk::DescriptorSet GetSet(int material_index) const {
    if (material_index < 0 || static_cast<std::size_t>(material_index) >= material_sets_.size()) {
      return *default_set_;
    }
    return *material_sets_[material_index];
  }

 private:
  std::uint32_t FindMemoryType(std::uint32_t type_filter, vk::MemoryPropertyFlags properties) const;
  void CreateFallbackTexture();
  void CreateLayoutAndPool(std::uint32_t material_count);
  void WriteMaterialSet(const vk::raii::DescriptorSet& set, const Material* material,
                       const Model& model);

  Device& device_;

  vk::raii::DescriptorSetLayout layout_ = nullptr;
  vk::raii::DescriptorPool pool_ = nullptr;

  // 1x1 white fallback used for every missing texture slot.
  vk::raii::Image fallback_image_ = nullptr;
  vk::raii::DeviceMemory fallback_memory_ = nullptr;
  vk::raii::ImageView fallback_view_ = nullptr;
  vk::raii::Sampler fallback_sampler_ = nullptr;

  std::vector<vk::raii::DescriptorSet> material_sets_;  // one per Material
  vk::raii::DescriptorSet default_set_ = nullptr;       // for primitives w/o material
};

// ---------------------------------------------------------------------------: Implementation

MaterialBindings::MaterialBindings(Device& device, Model& model) : device_(device) {
  CreateFallbackTexture();

  const std::uint32_t material_count = static_cast<std::uint32_t>(model.materials.size());
  CreateLayoutAndPool(material_count + 1);  // +1 reserved for the default set

  const auto& dev = device_.GetLogicalDevice();

  // Allocate one set per material plus the default set. vk::raii::DescriptorSets
  // returns a vector; we split the last element off as default_set_.
  std::vector<vk::DescriptorSetLayout> layouts(material_count + 1, *layout_);
  vk::DescriptorSetAllocateInfo alloc_info{
      .descriptorPool = *pool_,
      .descriptorSetCount = material_count + 1,
      .pSetLayouts = layouts.data(),
  };
  auto sets = vk::raii::DescriptorSets(dev, alloc_info);
  default_set_ = std::move(sets.back());
  sets.pop_back();
  material_sets_ = std::move(sets);

  for (std::uint32_t i = 0; i < material_count; ++i) {
    WriteMaterialSet(material_sets_[i], &model.materials[i], model);
  }
  WriteMaterialSet(default_set_, nullptr, model);
}

void MaterialBindings::CreateFallbackTexture() {
  const auto& dev = device_.GetLogicalDevice();

  // Host staging buffer with a single white pixel.
  const std::array<std::uint8_t, 4> pixel{0xFF, 0xFF, 0xFF, 0xFF};
  vk::BufferCreateInfo buffer_info{
      .size = pixel.size(),
      .usage = vk::BufferUsageFlagBits::eTransferSrc,
      .sharingMode = vk::SharingMode::eExclusive,
  };
  vk::raii::Buffer staging(dev, buffer_info);
  vk::MemoryRequirements staging_reqs = staging.getMemoryRequirements();
  vk::MemoryAllocateInfo staging_alloc{
      .allocationSize = staging_reqs.size,
      .memoryTypeIndex = FindMemoryType(
          staging_reqs.memoryTypeBits,
          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
  };
  vk::raii::DeviceMemory staging_mem(dev, staging_alloc);
  staging.bindMemory(*staging_mem, 0);
  void* mapped = staging_mem.mapMemory(0, pixel.size());
  std::memcpy(mapped, pixel.data(), pixel.size());
  staging_mem.unmapMemory();

  vk::ImageCreateInfo image_info{
      .imageType = vk::ImageType::e2D,
      .format = vk::Format::eR8G8B8A8Unorm,
      .extent = vk::Extent3D{1, 1, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = vk::SampleCountFlagBits::e1,
      .tiling = vk::ImageTiling::eOptimal,
      .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
      .sharingMode = vk::SharingMode::eExclusive,
      .initialLayout = vk::ImageLayout::eUndefined,
  };
  fallback_image_ = vk::raii::Image(dev, image_info);

  vk::MemoryRequirements reqs = fallback_image_.getMemoryRequirements();
  vk::MemoryAllocateInfo alloc_info{
      .allocationSize = reqs.size,
      .memoryTypeIndex = FindMemoryType(reqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
  };
  fallback_memory_ = vk::raii::DeviceMemory(dev, alloc_info);
  fallback_image_.bindMemory(*fallback_memory_, 0);

  // One-shot command buffer to transition + copy + transition.
  vk::CommandPoolCreateInfo pool_info{
      .flags = vk::CommandPoolCreateFlagBits::eTransient,
      .queueFamilyIndex = device_.GetGraphicsQueueFamily(),
  };
  vk::raii::CommandPool pool(dev, pool_info);
  vk::CommandBufferAllocateInfo cb_alloc{
      .commandPool = *pool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
  };
  auto cmds = vk::raii::CommandBuffers(dev, cb_alloc);
  auto& cmd = cmds.front();
  cmd.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

  vk::ImageMemoryBarrier2 to_dst{
      .srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe,
      .srcAccessMask = {},
      .dstStageMask = vk::PipelineStageFlagBits2::eTransfer,
      .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .oldLayout = vk::ImageLayout::eUndefined,
      .newLayout = vk::ImageLayout::eTransferDstOptimal,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = *fallback_image_,
      .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  vk::DependencyInfo dep_dst{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_dst};
  cmd.pipelineBarrier2(dep_dst);

  vk::BufferImageCopy region{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
      .imageOffset = {0, 0, 0},
      .imageExtent = {1, 1, 1},
  };
  cmd.copyBufferToImage(*staging, *fallback_image_, vk::ImageLayout::eTransferDstOptimal, region);

  vk::ImageMemoryBarrier2 to_shader{
      .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
      .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
      .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
      .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
      .oldLayout = vk::ImageLayout::eTransferDstOptimal,
      .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = *fallback_image_,
      .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  vk::DependencyInfo dep_shader{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &to_shader};
  cmd.pipelineBarrier2(dep_shader);
  cmd.end();

  vk::SubmitInfo submit{.commandBufferCount = 1, .pCommandBuffers = &*cmd};
  device_.GetGraphicsQueue().submit(submit);
  device_.GetGraphicsQueue().waitIdle();

  vk::ImageViewCreateInfo view_info{
      .image = *fallback_image_,
      .viewType = vk::ImageViewType::e2D,
      .format = vk::Format::eR8G8B8A8Unorm,
      .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };
  fallback_view_ = vk::raii::ImageView(dev, view_info);

  vk::SamplerCreateInfo sampler_info{
      .magFilter = vk::Filter::eNearest,
      .minFilter = vk::Filter::eNearest,
      .mipmapMode = vk::SamplerMipmapMode::eNearest,
      .addressModeU = vk::SamplerAddressMode::eRepeat,
      .addressModeV = vk::SamplerAddressMode::eRepeat,
      .addressModeW = vk::SamplerAddressMode::eRepeat,
      .maxLod = 1.0f,
  };
  fallback_sampler_ = vk::raii::Sampler(dev, sampler_info);
}

void MaterialBindings::CreateLayoutAndPool(std::uint32_t set_count) {
  const auto& dev = device_.GetLogicalDevice();

  std::array<vk::DescriptorSetLayoutBinding, 5> bindings{};
  for (std::uint32_t i = 0; i < bindings.size(); ++i) {
    bindings[i] = vk::DescriptorSetLayoutBinding{
        .binding = i,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
    };
  }
  vk::DescriptorSetLayoutCreateInfo layout_info{
      .bindingCount = static_cast<std::uint32_t>(bindings.size()),
      .pBindings = bindings.data(),
  };
  layout_ = vk::raii::DescriptorSetLayout(dev, layout_info);

  vk::DescriptorPoolSize pool_size{
      .type = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = set_count * 5,  // 5 bindings per set
  };
  vk::DescriptorPoolCreateInfo pool_info{
      .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
      .maxSets = set_count,
      .poolSizeCount = 1,
      .pPoolSizes = &pool_size,
  };
  pool_ = vk::raii::DescriptorPool(dev, pool_info);
}

void MaterialBindings::WriteMaterialSet(const vk::raii::DescriptorSet& set, const Material* material,
                                        const Model& model) {
  // Build an image-info entry for each of the 5 slots, pointing at either the
  // material's real texture (when the index is valid) or the fallback. The
  // shader checks the push-constant `*_tex_set` flag to decide whether to
  // actually use the sampled value, so binding the fallback everywhere is
  // always safe.
  auto pick = [&](int idx) -> std::pair<vk::Sampler, vk::ImageView> {
    if (material != nullptr && idx >= 0 && idx < static_cast<int>(model.textures.size())) {
      const auto& tex = model.textures[idx];
      return {*tex.sampler, *tex.view};
    }
    return {*fallback_sampler_, *fallback_view_};
  };

  std::array<int, 5> slots{
      material != nullptr ? material->base_color_texture : -1,
      material != nullptr ? material->metallic_roughness_texture : -1,
      material != nullptr ? material->normal_texture : -1,
      material != nullptr ? material->occlusion_texture : -1,
      material != nullptr ? material->emissive_texture : -1,
  };

  std::array<vk::DescriptorImageInfo, 5> infos{};
  std::array<vk::WriteDescriptorSet, 5> writes{};
  for (std::size_t i = 0; i < slots.size(); ++i) {
    auto [sampler, view] = pick(slots[i]);
    infos[i] = vk::DescriptorImageInfo{
        .sampler = sampler,
        .imageView = view,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    writes[i] = vk::WriteDescriptorSet{
        .dstSet = *set,
        .dstBinding = static_cast<std::uint32_t>(i),
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &infos[i],
    };
  }
  device_.GetLogicalDevice().updateDescriptorSets(writes, nullptr);
}

std::uint32_t MaterialBindings::FindMemoryType(std::uint32_t type_filter,
                                               vk::MemoryPropertyFlags properties) const {
  auto props = device_.GetPhysicalDevice().getMemoryProperties();
  for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    const bool type_ok = (type_filter & (1u << i)) != 0;
    const bool props_ok = (props.memoryTypes[i].propertyFlags & properties) == properties;
    if (type_ok && props_ok) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type for material bindings");
}

}  // namespace engine
