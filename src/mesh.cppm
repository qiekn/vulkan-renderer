module;

#include <glm/glm.hpp>

export module engine.mesh;

import vulkan;
import std;
import engine.device;

namespace engine {

// ---------------------------------------------------------------------------: Vertex

// Position + normal only for now. UV / tangent land in the Loading_Models chapter
// when we pull real glTF meshes with texture maps.
export struct Vertex {
  glm::vec3 position;
  glm::vec3 normal;
};

// ---------------------------------------------------------------------------: Mesh

// Holds host-visible vertex and index buffers. Good enough for small authored
// meshes (cube, sphere); when we start importing heavy assets we'll swap in
// device-local buffers fed by a staging buffer.
export class Mesh {
 public:
  Mesh(Device& device,
       std::span<const Vertex> vertices,
       std::span<const std::uint32_t> indices);

  Mesh(const Mesh&) = delete;
  Mesh& operator=(const Mesh&) = delete;
  Mesh(Mesh&&) = default;
  Mesh& operator=(Mesh&&) = default;

  const vk::raii::Buffer& GetVertexBuffer() const { return vertex_buffer_; }
  const vk::raii::Buffer& GetIndexBuffer() const { return index_buffer_; }
  std::uint32_t GetIndexCount() const { return index_count_; }

  static Mesh CreateCube(Device& device, float size = 1.0f);

  static vk::VertexInputBindingDescription GetBindingDescription();
  static std::array<vk::VertexInputAttributeDescription, 2> GetAttributeDescriptions();

 private:
  std::uint32_t FindMemoryType(std::uint32_t type_filter, vk::MemoryPropertyFlags properties) const;

  void CreateBuffer(vk::DeviceSize size,
                    vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties,
                    const void* data,
                    vk::raii::Buffer& out_buffer,
                    vk::raii::DeviceMemory& out_memory);

  Device* device_;
  vk::raii::Buffer vertex_buffer_ = nullptr;
  vk::raii::DeviceMemory vertex_memory_ = nullptr;
  vk::raii::Buffer index_buffer_ = nullptr;
  vk::raii::DeviceMemory index_memory_ = nullptr;
  std::uint32_t index_count_ = 0;
};

// ---------------------------------------------------------------------------: Implementation

Mesh::Mesh(Device& device,
           std::span<const Vertex> vertices,
           std::span<const std::uint32_t> indices)
    : device_(&device), index_count_(static_cast<std::uint32_t>(indices.size())) {
  CreateBuffer(vertices.size_bytes(),
               vk::BufferUsageFlagBits::eVertexBuffer,
               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
               vertices.data(), vertex_buffer_, vertex_memory_);
  CreateBuffer(indices.size_bytes(),
               vk::BufferUsageFlagBits::eIndexBuffer,
               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
               indices.data(), index_buffer_, index_memory_);
}

void Mesh::CreateBuffer(vk::DeviceSize size,
                        vk::BufferUsageFlags usage,
                        vk::MemoryPropertyFlags properties,
                        const void* data,
                        vk::raii::Buffer& out_buffer,
                        vk::raii::DeviceMemory& out_memory) {
  const auto& dev = device_->GetLogicalDevice();

  vk::BufferCreateInfo buffer_info{
      .size = size,
      .usage = usage,
      .sharingMode = vk::SharingMode::eExclusive,
  };
  out_buffer = vk::raii::Buffer(dev, buffer_info);

  vk::MemoryRequirements reqs = out_buffer.getMemoryRequirements();
  vk::MemoryAllocateInfo alloc_info{
      .allocationSize = reqs.size,
      .memoryTypeIndex = FindMemoryType(reqs.memoryTypeBits, properties),
  };
  out_memory = vk::raii::DeviceMemory(dev, alloc_info);
  out_buffer.bindMemory(*out_memory, 0);

  void* mapped = out_memory.mapMemory(0, size);
  std::memcpy(mapped, data, static_cast<std::size_t>(size));
  out_memory.unmapMemory();
}

std::uint32_t Mesh::FindMemoryType(std::uint32_t type_filter, vk::MemoryPropertyFlags properties) const {
  auto props = device_->GetPhysicalDevice().getMemoryProperties();
  for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
    bool type_ok = (type_filter & (1u << i)) != 0;
    bool props_ok = (props.memoryTypes[i].propertyFlags & properties) == properties;
    if (type_ok && props_ok) {
      return i;
    }
  }
  throw std::runtime_error("Failed to find suitable memory type for mesh buffer");
}

vk::VertexInputBindingDescription Mesh::GetBindingDescription() {
  return vk::VertexInputBindingDescription{
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = vk::VertexInputRate::eVertex,
  };
}

std::array<vk::VertexInputAttributeDescription, 2> Mesh::GetAttributeDescriptions() {
  return {
      vk::VertexInputAttributeDescription{
          .location = 0,
          .binding = 0,
          .format = vk::Format::eR32G32B32Sfloat,
          .offset = offsetof(Vertex, position),
      },
      vk::VertexInputAttributeDescription{
          .location = 1,
          .binding = 0,
          .format = vk::Format::eR32G32B32Sfloat,
          .offset = offsetof(Vertex, normal),
      },
  };
}

Mesh Mesh::CreateCube(Device& device, float size) {
  const float h = size * 0.5f;
  // 24 vertices: each face contributes 4 with its own face-constant normal so
  // per-face lighting stays flat instead of getting smoothed across edges.
  const std::array<Vertex, 24> vertices = {{
      // +X face
      {{ h, -h, -h}, { 1.0f,  0.0f,  0.0f}},
      {{ h,  h, -h}, { 1.0f,  0.0f,  0.0f}},
      {{ h,  h,  h}, { 1.0f,  0.0f,  0.0f}},
      {{ h, -h,  h}, { 1.0f,  0.0f,  0.0f}},
      // -X face
      {{-h, -h,  h}, {-1.0f,  0.0f,  0.0f}},
      {{-h,  h,  h}, {-1.0f,  0.0f,  0.0f}},
      {{-h,  h, -h}, {-1.0f,  0.0f,  0.0f}},
      {{-h, -h, -h}, {-1.0f,  0.0f,  0.0f}},
      // +Y face
      {{-h,  h, -h}, { 0.0f,  1.0f,  0.0f}},
      {{-h,  h,  h}, { 0.0f,  1.0f,  0.0f}},
      {{ h,  h,  h}, { 0.0f,  1.0f,  0.0f}},
      {{ h,  h, -h}, { 0.0f,  1.0f,  0.0f}},
      // -Y face
      {{-h, -h,  h}, { 0.0f, -1.0f,  0.0f}},
      {{-h, -h, -h}, { 0.0f, -1.0f,  0.0f}},
      {{ h, -h, -h}, { 0.0f, -1.0f,  0.0f}},
      {{ h, -h,  h}, { 0.0f, -1.0f,  0.0f}},
      // +Z face
      {{ h, -h,  h}, { 0.0f,  0.0f,  1.0f}},
      {{ h,  h,  h}, { 0.0f,  0.0f,  1.0f}},
      {{-h,  h,  h}, { 0.0f,  0.0f,  1.0f}},
      {{-h, -h,  h}, { 0.0f,  0.0f,  1.0f}},
      // -Z face
      {{-h, -h, -h}, { 0.0f,  0.0f, -1.0f}},
      {{-h,  h, -h}, { 0.0f,  0.0f, -1.0f}},
      {{ h,  h, -h}, { 0.0f,  0.0f, -1.0f}},
      {{ h, -h, -h}, { 0.0f,  0.0f, -1.0f}},
  }};

  const std::array<std::uint32_t, 36> indices = {
       0,  1,  2,   0,  2,  3,   // +X
       4,  5,  6,   4,  6,  7,   // -X
       8,  9, 10,   8, 10, 11,   // +Y
      12, 13, 14,  12, 14, 15,   // -Y
      16, 17, 18,  16, 18, 19,   // +Z
      20, 21, 22,  20, 22, 23,   // -Z
  };

  return Mesh(device, vertices, indices);
}

}  // namespace engine
