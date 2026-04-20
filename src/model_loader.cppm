module;

#include <cassert>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <tiny_gltf.h>

export module engine.model_loader;

import vulkan;
import std;
import engine.device;
import engine.model;

namespace engine {

// ---------------------------------------------------------------------------: Vulkan upload helpers

namespace {

std::uint32_t FindMemoryType(const Device& device, std::uint32_t type_filter,
                             vk::MemoryPropertyFlags properties) {
  auto props = device.GetPhysicalDevice().getMemoryProperties();
  for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    const bool type_ok = (type_filter & (1u << i)) != 0;
    const bool props_ok = (props.memoryTypes[i].propertyFlags & properties) == properties;
    if (type_ok && props_ok) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type");
}

// Single-shot command recorder: allocates one primary buffer from the provided
// pool, begins it on construction, submits and waits on Finish(). Everything
// here runs synchronously — fine for asset loading during startup but not for
// streaming.
class OneTimeCommands {
 public:
  OneTimeCommands(const Device& device, const vk::raii::CommandPool& pool) : device_(device) {
    vk::CommandBufferAllocateInfo info{
        .commandPool = *pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    auto buffers = vk::raii::CommandBuffers(device.GetLogicalDevice(), info);
    cmd_ = std::move(buffers.front());
    cmd_.begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
  }

  vk::raii::CommandBuffer& Cmd() { return cmd_; }

  void Finish() {
    cmd_.end();
    vk::SubmitInfo submit{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd_,
    };
    device_.GetGraphicsQueue().submit(submit);
    device_.GetGraphicsQueue().waitIdle();
  }

 private:
  const Device& device_;
  vk::raii::CommandBuffer cmd_ = nullptr;
};

struct StagingBuffer {
  vk::raii::Buffer buffer = nullptr;
  vk::raii::DeviceMemory memory = nullptr;
};

StagingBuffer CreateStagingBuffer(const Device& device, vk::DeviceSize size, const void* data) {
  StagingBuffer staging;
  vk::BufferCreateInfo buffer_info{
      .size = size,
      .usage = vk::BufferUsageFlagBits::eTransferSrc,
      .sharingMode = vk::SharingMode::eExclusive,
  };
  staging.buffer = vk::raii::Buffer(device.GetLogicalDevice(), buffer_info);

  vk::MemoryRequirements reqs = staging.buffer.getMemoryRequirements();
  vk::MemoryAllocateInfo alloc_info{
      .allocationSize = reqs.size,
      .memoryTypeIndex = FindMemoryType(
          device, reqs.memoryTypeBits,
          vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
  };
  staging.memory = vk::raii::DeviceMemory(device.GetLogicalDevice(), alloc_info);
  staging.buffer.bindMemory(*staging.memory, 0);

  if (data != nullptr && size > 0) {
    void* mapped = staging.memory.mapMemory(0, size);
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    staging.memory.unmapMemory();
  }
  return staging;
}

// Allocates a device-local buffer and copies `data` into it via a staging
// buffer + one-time copy. Device-local buffers win on every access after the
// upload, which matters once we're walking thousands of vertices per frame.
void UploadDeviceLocalBuffer(const Device& device, const vk::raii::CommandPool& pool,
                             const void* data, vk::DeviceSize size,
                             vk::BufferUsageFlags usage,
                             vk::raii::Buffer& out_buffer,
                             vk::raii::DeviceMemory& out_memory) {
  auto staging = CreateStagingBuffer(device, size, data);

  vk::BufferCreateInfo buffer_info{
      .size = size,
      .usage = usage | vk::BufferUsageFlagBits::eTransferDst,
      .sharingMode = vk::SharingMode::eExclusive,
  };
  out_buffer = vk::raii::Buffer(device.GetLogicalDevice(), buffer_info);

  vk::MemoryRequirements reqs = out_buffer.getMemoryRequirements();
  vk::MemoryAllocateInfo alloc_info{
      .allocationSize = reqs.size,
      .memoryTypeIndex = FindMemoryType(device, reqs.memoryTypeBits,
                                        vk::MemoryPropertyFlagBits::eDeviceLocal),
  };
  out_memory = vk::raii::DeviceMemory(device.GetLogicalDevice(), alloc_info);
  out_buffer.bindMemory(*out_memory, 0);

  OneTimeCommands cmd(device, pool);
  vk::BufferCopy region{.srcOffset = 0, .dstOffset = 0, .size = size};
  cmd.Cmd().copyBuffer(*staging.buffer, *out_buffer, region);
  cmd.Finish();
}

// Transitions `image` from `old_layout` → `new_layout`. Picks access masks
// based on the transitions we actually use (undefined → transferDst →
// shaderReadOnly); throw if something else shows up so we don't silently
// forget to cover a new case.
void TransitionImageLayout(vk::raii::CommandBuffer& cmd, vk::Image image,
                           vk::ImageLayout old_layout, vk::ImageLayout new_layout,
                           std::uint32_t mip_levels) {
  vk::ImageMemoryBarrier2 barrier{
      .oldLayout = old_layout,
      .newLayout = new_layout,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = image,
      .subresourceRange =
          vk::ImageSubresourceRange{
              .aspectMask = vk::ImageAspectFlagBits::eColor,
              .baseMipLevel = 0,
              .levelCount = mip_levels,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
  };

  if (old_layout == vk::ImageLayout::eUndefined &&
      new_layout == vk::ImageLayout::eTransferDstOptimal) {
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    barrier.srcAccessMask = {};
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eTransfer;
    barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
  } else if (old_layout == vk::ImageLayout::eTransferDstOptimal &&
             new_layout == vk::ImageLayout::eShaderReadOnlyOptimal) {
    barrier.srcStageMask = vk::PipelineStageFlagBits2::eTransfer;
    barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
    barrier.dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader;
    barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead;
  } else {
    throw std::runtime_error("Unsupported image layout transition in model loader");
  }

  vk::DependencyInfo dep{
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
  };
  cmd.pipelineBarrier2(dep);
}

void CopyBufferToImage(vk::raii::CommandBuffer& cmd, vk::Buffer buffer, vk::Image image,
                       std::uint32_t width, std::uint32_t height) {
  vk::BufferImageCopy region{
      .bufferOffset = 0,
      .bufferRowLength = 0,
      .bufferImageHeight = 0,
      .imageSubresource =
          vk::ImageSubresourceLayers{
              .aspectMask = vk::ImageAspectFlagBits::eColor,
              .mipLevel = 0,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
      .imageOffset = vk::Offset3D{0, 0, 0},
      .imageExtent = vk::Extent3D{width, height, 1},
  };
  cmd.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
}

// Uploads an RGBA8 byte blob into a new device-local vk::Image and returns
// the full ModelTexture (image + memory + view + sampler). Format is chosen
// by the caller so base-color / emissive textures go through sRGB while
// data textures (normals, metallic/roughness, occlusion) stay linear.
ModelTexture CreateTextureFromPixels(const Device& device, const vk::raii::CommandPool& pool,
                                     const unsigned char* pixels, std::uint32_t width,
                                     std::uint32_t height, vk::Format format) {
  const vk::DeviceSize size = static_cast<vk::DeviceSize>(width) * height * 4;
  auto staging = CreateStagingBuffer(device, size, pixels);

  ModelTexture tex;
  tex.width = width;
  tex.height = height;
  tex.mip_levels = 1;

  vk::ImageCreateInfo image_info{
      .imageType = vk::ImageType::e2D,
      .format = format,
      .extent = vk::Extent3D{width, height, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = vk::SampleCountFlagBits::e1,
      .tiling = vk::ImageTiling::eOptimal,
      .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
      .sharingMode = vk::SharingMode::eExclusive,
      .initialLayout = vk::ImageLayout::eUndefined,
  };
  tex.image = vk::raii::Image(device.GetLogicalDevice(), image_info);

  vk::MemoryRequirements reqs = tex.image.getMemoryRequirements();
  vk::MemoryAllocateInfo alloc_info{
      .allocationSize = reqs.size,
      .memoryTypeIndex = FindMemoryType(device, reqs.memoryTypeBits,
                                        vk::MemoryPropertyFlagBits::eDeviceLocal),
  };
  tex.memory = vk::raii::DeviceMemory(device.GetLogicalDevice(), alloc_info);
  tex.image.bindMemory(*tex.memory, 0);

  {
    OneTimeCommands cmd(device, pool);
    TransitionImageLayout(cmd.Cmd(), *tex.image, vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal, 1);
    CopyBufferToImage(cmd.Cmd(), *staging.buffer, *tex.image, width, height);
    TransitionImageLayout(cmd.Cmd(), *tex.image, vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal, 1);
    cmd.Finish();
  }

  vk::ImageViewCreateInfo view_info{
      .image = *tex.image,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .components = {},
      .subresourceRange =
          vk::ImageSubresourceRange{
              .aspectMask = vk::ImageAspectFlagBits::eColor,
              .baseMipLevel = 0,
              .levelCount = 1,
              .baseArrayLayer = 0,
              .layerCount = 1,
          },
  };
  tex.view = vk::raii::ImageView(device.GetLogicalDevice(), view_info);

  vk::SamplerCreateInfo sampler_info{
      .magFilter = vk::Filter::eLinear,
      .minFilter = vk::Filter::eLinear,
      .mipmapMode = vk::SamplerMipmapMode::eLinear,
      .addressModeU = vk::SamplerAddressMode::eRepeat,
      .addressModeV = vk::SamplerAddressMode::eRepeat,
      .addressModeW = vk::SamplerAddressMode::eRepeat,
      .mipLodBias = 0.0f,
      .anisotropyEnable = vk::False,
      .maxAnisotropy = 1.0f,
      .compareEnable = vk::False,
      .compareOp = vk::CompareOp::eNever,
      .minLod = 0.0f,
      .maxLod = 1.0f,
      .borderColor = vk::BorderColor::eIntOpaqueBlack,
      .unnormalizedCoordinates = vk::False,
  };
  tex.sampler = vk::raii::Sampler(device.GetLogicalDevice(), sampler_info);

  return tex;
}

// ---------------------------------------------------------------------------: glTF accessor helpers

// Returns a typed pointer to the first element the accessor describes within
// its buffer view. Callers still have to respect `byteStride` when walking
// past element 0 — glTF lets attributes interleave into the same buffer view.
const std::byte* AccessorData(const tinygltf::Model& gltf, int accessor_index, std::size_t& stride) {
  const tinygltf::Accessor& accessor = gltf.accessors[accessor_index];
  const tinygltf::BufferView& view = gltf.bufferViews[accessor.bufferView];
  const tinygltf::Buffer& buffer = gltf.buffers[view.buffer];
  const std::size_t component_size = tinygltf::GetComponentSizeInBytes(accessor.componentType);
  const std::size_t element_count = tinygltf::GetNumComponentsInType(accessor.type);
  stride = view.byteStride != 0 ? view.byteStride : component_size * element_count;
  const std::size_t base = view.byteOffset + accessor.byteOffset;
  return reinterpret_cast<const std::byte*>(buffer.data.data()) + base;
}

// Copies N indices out of an accessor, widening 8/16-bit sources to uint32_t
// so the draw can use vk::IndexType::eUint32 uniformly.
std::vector<std::uint32_t> ReadIndices(const tinygltf::Model& gltf, int accessor_index) {
  const tinygltf::Accessor& accessor = gltf.accessors[accessor_index];
  std::size_t stride = 0;
  const std::byte* ptr = AccessorData(gltf, accessor_index, stride);

  std::vector<std::uint32_t> indices;
  indices.reserve(accessor.count);
  switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
      for (std::size_t i = 0; i < accessor.count; ++i) {
        indices.push_back(*reinterpret_cast<const std::uint8_t*>(ptr + i * stride));
      }
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
      for (std::size_t i = 0; i < accessor.count; ++i) {
        indices.push_back(*reinterpret_cast<const std::uint16_t*>(ptr + i * stride));
      }
      break;
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
      for (std::size_t i = 0; i < accessor.count; ++i) {
        indices.push_back(*reinterpret_cast<const std::uint32_t*>(ptr + i * stride));
      }
      break;
    default:
      throw std::runtime_error("Unsupported glTF index component type");
  }
  return indices;
}

// Pulls one vec3 attribute (positions, normals, ...) into a flat std::vector.
// Returns empty if the attribute isn't present on the primitive so the caller
// can fall back to zero/default values without a second lookup.
std::vector<glm::vec3> ReadVec3(const tinygltf::Model& gltf, const tinygltf::Primitive& prim,
                                const char* attr) {
  auto it = prim.attributes.find(attr);
  if (it == prim.attributes.end()) {
    return {};
  }
  const tinygltf::Accessor& accessor = gltf.accessors[it->second];
  std::size_t stride = 0;
  const std::byte* ptr = AccessorData(gltf, it->second, stride);

  std::vector<glm::vec3> out;
  out.reserve(accessor.count);
  for (std::size_t i = 0; i < accessor.count; ++i) {
    const float* f = reinterpret_cast<const float*>(ptr + i * stride);
    out.emplace_back(f[0], f[1], f[2]);
  }
  return out;
}

std::vector<glm::vec2> ReadVec2(const tinygltf::Model& gltf, const tinygltf::Primitive& prim,
                                const char* attr) {
  auto it = prim.attributes.find(attr);
  if (it == prim.attributes.end()) {
    return {};
  }
  const tinygltf::Accessor& accessor = gltf.accessors[it->second];
  std::size_t stride = 0;
  const std::byte* ptr = AccessorData(gltf, it->second, stride);

  std::vector<glm::vec2> out;
  out.reserve(accessor.count);
  for (std::size_t i = 0; i < accessor.count; ++i) {
    const float* f = reinterpret_cast<const float*>(ptr + i * stride);
    out.emplace_back(f[0], f[1]);
  }
  return out;
}

std::vector<glm::vec4> ReadVec4(const tinygltf::Model& gltf, const tinygltf::Primitive& prim,
                                const char* attr) {
  auto it = prim.attributes.find(attr);
  if (it == prim.attributes.end()) {
    return {};
  }
  const tinygltf::Accessor& accessor = gltf.accessors[it->second];
  std::size_t stride = 0;
  const std::byte* ptr = AccessorData(gltf, it->second, stride);

  std::vector<glm::vec4> out;
  out.reserve(accessor.count);
  for (std::size_t i = 0; i < accessor.count; ++i) {
    const float* f = reinterpret_cast<const float*>(ptr + i * stride);
    out.emplace_back(f[0], f[1], f[2], f[3]);
  }
  return out;
}

// Same shape as ReadVec4 but with an animation-sampler fallback: scalars
// widen to vec4(x, 0, 0, 0) and vec3s widen to vec4(xyz, 0). Keeps the single
// outputs container regardless of channel path.
std::vector<glm::vec4> ReadSamplerOutputs(const tinygltf::Model& gltf, int accessor_index) {
  const tinygltf::Accessor& accessor = gltf.accessors[accessor_index];
  std::size_t stride = 0;
  const std::byte* ptr = AccessorData(gltf, accessor_index, stride);
  const std::size_t components = tinygltf::GetNumComponentsInType(accessor.type);

  std::vector<glm::vec4> out;
  out.reserve(accessor.count);
  for (std::size_t i = 0; i < accessor.count; ++i) {
    const float* f = reinterpret_cast<const float*>(ptr + i * stride);
    glm::vec4 v(0.0f);
    for (std::size_t c = 0; c < components && c < 4; ++c) {
      v[static_cast<int>(c)] = f[c];
    }
    out.push_back(v);
  }
  return out;
}

std::vector<float> ReadSamplerInputs(const tinygltf::Model& gltf, int accessor_index) {
  const tinygltf::Accessor& accessor = gltf.accessors[accessor_index];
  std::size_t stride = 0;
  const std::byte* ptr = AccessorData(gltf, accessor_index, stride);

  std::vector<float> out;
  out.reserve(accessor.count);
  for (std::size_t i = 0; i < accessor.count; ++i) {
    out.push_back(*reinterpret_cast<const float*>(ptr + i * stride));
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------: Public entry point

// Parses `path` with tinygltf and produces a fully uploaded Model. Creates a
// transient graphics command pool for the one-time copies; the pool dies at
// function exit once every submission has drained on the graphics queue.
export Model LoadGltfModel(Device& device, const std::filesystem::path& path) {
  tinygltf::Model gltf;
  tinygltf::TinyGLTF loader;
  std::string err, warn;

  const std::string path_str = path.string();
  const bool is_binary = path.extension() == ".glb";
  const bool ok = is_binary ? loader.LoadBinaryFromFile(&gltf, &err, &warn, path_str)
                            : loader.LoadASCIIFromFile(&gltf, &err, &warn, path_str);
  if (!warn.empty()) {
    std::cerr << "[gltf] warning loading " << path_str << ": " << warn << '\n';
  }
  if (!err.empty()) {
    std::cerr << "[gltf] error loading " << path_str << ": " << err << '\n';
  }
  if (!ok) {
    throw std::runtime_error("Failed to load glTF file: " + path_str);
  }

  vk::CommandPoolCreateInfo pool_info{
      .flags = vk::CommandPoolCreateFlagBits::eTransient,
      .queueFamilyIndex = device.GetGraphicsQueueFamily(),
  };
  vk::raii::CommandPool pool(device.GetLogicalDevice(), pool_info);

  Model model;

  // Textures: each glTF texture references an image + sampler; we fold both
  // into a single ModelTexture and remember whether it's used in an sRGB slot
  // so we pick the right format. Per glTF spec only baseColor and emissive
  // need sRGB; normals / MR / occlusion stay linear. We can't tell from the
  // image alone, so decide at material-link time after this pass.
  std::vector<bool> texture_is_srgb(gltf.textures.size(), false);
  for (const auto& material : gltf.materials) {
    const int base = material.pbrMetallicRoughness.baseColorTexture.index;
    const int emissive = material.emissiveTexture.index;
    if (base >= 0 && base < static_cast<int>(texture_is_srgb.size())) {
      texture_is_srgb[base] = true;
    }
    if (emissive >= 0 && emissive < static_cast<int>(texture_is_srgb.size())) {
      texture_is_srgb[emissive] = true;
    }
  }

  model.textures.reserve(gltf.textures.size());
  for (std::size_t i = 0; i < gltf.textures.size(); ++i) {
    const tinygltf::Texture& tex = gltf.textures[i];
    if (tex.source < 0 || tex.source >= static_cast<int>(gltf.images.size())) {
      continue;
    }
    const tinygltf::Image& image = gltf.images[tex.source];
    if (image.image.empty() || image.width <= 0 || image.height <= 0) {
      continue;
    }

    // tinygltf decodes to 1/2/3/4-channel 8-bit; we always want RGBA8 for the
    // shader, so expand on the host side when it's short a channel.
    std::vector<unsigned char> rgba;
    const unsigned char* src = image.image.data();
    if (image.component != 4) {
      rgba.resize(static_cast<std::size_t>(image.width) * image.height * 4, 0xFF);
      for (int p = 0; p < image.width * image.height; ++p) {
        for (int c = 0; c < image.component && c < 4; ++c) {
          rgba[p * 4 + c] = image.image[p * image.component + c];
        }
      }
      src = rgba.data();
    }

    const vk::Format format = texture_is_srgb[i] ? vk::Format::eR8G8B8A8Srgb
                                                  : vk::Format::eR8G8B8A8Unorm;
    model.textures.push_back(CreateTextureFromPixels(
        device, pool, src, static_cast<std::uint32_t>(image.width),
        static_cast<std::uint32_t>(image.height), format));
  }

  // Materials: copy factors and texture indices. tinygltf's `index` into our
  // texture list maps 1:1 because we preserve order.
  model.materials.reserve(gltf.materials.size());
  for (const tinygltf::Material& gm : gltf.materials) {
    Material mat;
    const auto& pbr = gm.pbrMetallicRoughness;
    if (pbr.baseColorFactor.size() == 4) {
      mat.base_color_factor = glm::vec4(
          static_cast<float>(pbr.baseColorFactor[0]),
          static_cast<float>(pbr.baseColorFactor[1]),
          static_cast<float>(pbr.baseColorFactor[2]),
          static_cast<float>(pbr.baseColorFactor[3]));
    }
    mat.metallic_factor = static_cast<float>(pbr.metallicFactor);
    mat.roughness_factor = static_cast<float>(pbr.roughnessFactor);
    if (gm.emissiveFactor.size() == 3) {
      mat.emissive_factor = glm::vec3(
          static_cast<float>(gm.emissiveFactor[0]),
          static_cast<float>(gm.emissiveFactor[1]),
          static_cast<float>(gm.emissiveFactor[2]));
    }
    mat.base_color_texture = pbr.baseColorTexture.index;
    mat.metallic_roughness_texture = pbr.metallicRoughnessTexture.index;
    mat.normal_texture = gm.normalTexture.index;
    mat.occlusion_texture = gm.occlusionTexture.index;
    mat.emissive_texture = gm.emissiveTexture.index;
    model.materials.push_back(mat);
  }

  // Nodes: two passes — first materialize every node so parent pointers can
  // resolve, then link children. The glTF node index is preserved in
  // model.nodes[i] so we can cross-reference by index later (animations).
  model.nodes.reserve(gltf.nodes.size());
  for (std::size_t i = 0; i < gltf.nodes.size(); ++i) {
    auto node = std::make_unique<Node>();
    node->index = static_cast<std::uint32_t>(i);
    node->name = gltf.nodes[i].name;

    const auto& gn = gltf.nodes[i];
    if (gn.translation.size() == 3) {
      node->translation = glm::vec3(
          static_cast<float>(gn.translation[0]),
          static_cast<float>(gn.translation[1]),
          static_cast<float>(gn.translation[2]));
    }
    if (gn.rotation.size() == 4) {
      // glTF stores quats as (x, y, z, w); GLM expects (w, x, y, z).
      node->rotation = glm::quat(
          static_cast<float>(gn.rotation[3]),
          static_cast<float>(gn.rotation[0]),
          static_cast<float>(gn.rotation[1]),
          static_cast<float>(gn.rotation[2]));
    }
    if (gn.scale.size() == 3) {
      node->scale = glm::vec3(
          static_cast<float>(gn.scale[0]),
          static_cast<float>(gn.scale[1]),
          static_cast<float>(gn.scale[2]));
    }
    if (gn.matrix.size() == 16) {
      node->matrix = glm::make_mat4(gn.matrix.data());
    }

    model.nodes.push_back(std::move(node));
  }

  for (std::size_t i = 0; i < gltf.nodes.size(); ++i) {
    const auto& gn = gltf.nodes[i];
    for (int child_idx : gn.children) {
      if (child_idx < 0 || child_idx >= static_cast<int>(model.nodes.size())) {
        continue;
      }
      model.nodes[child_idx]->parent = model.nodes[i].get();
      model.nodes[i]->children.push_back(model.nodes[child_idx].get());
    }
  }

  // Root nodes: everything with no parent. Follows the scene default if the
  // glTF declares one so we honour which nodes the artist intended to expose.
  if (gltf.defaultScene >= 0 && gltf.defaultScene < static_cast<int>(gltf.scenes.size())) {
    for (int idx : gltf.scenes[gltf.defaultScene].nodes) {
      if (idx >= 0 && idx < static_cast<int>(model.nodes.size())) {
        model.root_nodes.push_back(model.nodes[idx].get());
      }
    }
  } else {
    for (auto& node : model.nodes) {
      if (node->parent == nullptr) {
        model.root_nodes.push_back(node.get());
      }
    }
  }

  // Primitives: for every node that has a mesh, interleave each primitive's
  // attributes into a ModelVertex buffer and upload vertex + index arrays to
  // device-local memory.
  for (std::size_t i = 0; i < gltf.nodes.size(); ++i) {
    const auto& gn = gltf.nodes[i];
    if (gn.mesh < 0 || gn.mesh >= static_cast<int>(gltf.meshes.size())) {
      continue;
    }
    const tinygltf::Mesh& mesh = gltf.meshes[gn.mesh];

    for (const tinygltf::Primitive& prim : mesh.primitives) {
      const auto positions = ReadVec3(gltf, prim, "POSITION");
      const auto normals = ReadVec3(gltf, prim, "NORMAL");
      const auto uvs = ReadVec2(gltf, prim, "TEXCOORD_0");
      const auto tangents = ReadVec4(gltf, prim, "TANGENT");
      if (positions.empty() || prim.indices < 0) {
        continue;
      }

      std::vector<ModelVertex> vertices;
      vertices.reserve(positions.size());
      for (std::size_t v = 0; v < positions.size(); ++v) {
        ModelVertex vert{};
        vert.position = positions[v];
        vert.normal = v < normals.size() ? normals[v] : glm::vec3(0.0f, 1.0f, 0.0f);
        vert.uv = v < uvs.size() ? uvs[v] : glm::vec2(0.0f);
        // Tangent.w = +1 by default so the shader's bitangent reconstruction
        // (cross(N, T.xyz) * T.w) stays right-handed when we lack real data.
        vert.tangent = v < tangents.size() ? tangents[v] : glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        vertices.push_back(vert);
      }
      const auto indices = ReadIndices(gltf, prim.indices);

      MeshPrimitive out_prim;
      out_prim.index_count = static_cast<std::uint32_t>(indices.size());
      out_prim.material_index = prim.material;
      UploadDeviceLocalBuffer(device, pool, vertices.data(),
                              vertices.size() * sizeof(ModelVertex),
                              vk::BufferUsageFlagBits::eVertexBuffer,
                              out_prim.vertex_buffer, out_prim.vertex_memory);
      UploadDeviceLocalBuffer(device, pool, indices.data(),
                              indices.size() * sizeof(std::uint32_t),
                              vk::BufferUsageFlagBits::eIndexBuffer,
                              out_prim.index_buffer, out_prim.index_memory);
      model.nodes[i]->primitives.push_back(std::move(out_prim));
    }
  }

  // Animations: remap target node indices to our owned Node pointers so the
  // runtime playback loop can mutate TRS directly.
  model.animations.reserve(gltf.animations.size());
  for (const tinygltf::Animation& ga : gltf.animations) {
    Animation anim;
    anim.name = ga.name;

    anim.samplers.reserve(ga.samplers.size());
    for (const auto& gs : ga.samplers) {
      AnimationSampler sampler;
      if (gs.interpolation == "LINEAR") {
        sampler.interpolation = AnimationSampler::Interpolation::Linear;
      } else if (gs.interpolation == "STEP") {
        sampler.interpolation = AnimationSampler::Interpolation::Step;
      } else if (gs.interpolation == "CUBICSPLINE") {
        sampler.interpolation = AnimationSampler::Interpolation::CubicSpline;
      }
      sampler.inputs = ReadSamplerInputs(gltf, gs.input);
      sampler.outputs = ReadSamplerOutputs(gltf, gs.output);
      if (!sampler.inputs.empty()) {
        anim.start = std::min(anim.start, sampler.inputs.front());
        anim.end = std::max(anim.end, sampler.inputs.back());
      }
      anim.samplers.push_back(std::move(sampler));
    }

    anim.channels.reserve(ga.channels.size());
    for (const auto& gc : ga.channels) {
      AnimationChannel channel;
      channel.sampler_index = static_cast<std::uint32_t>(gc.sampler);
      if (gc.target_node >= 0 && gc.target_node < static_cast<int>(model.nodes.size())) {
        channel.node = model.nodes[gc.target_node].get();
      }
      if (gc.target_path == "translation") {
        channel.path = AnimationChannel::Path::Translation;
      } else if (gc.target_path == "rotation") {
        channel.path = AnimationChannel::Path::Rotation;
      } else if (gc.target_path == "scale") {
        channel.path = AnimationChannel::Path::Scale;
      } else {
        continue;  // weights / unsupported paths skipped for now
      }
      anim.channels.push_back(channel);
    }

    if (anim.start > anim.end) {
      anim.start = 0.0f;
      anim.end = 0.0f;
    }
    model.animations.push_back(std::move(anim));
  }

  return model;
}

}  // namespace engine
