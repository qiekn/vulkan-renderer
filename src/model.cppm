module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

export module engine.model;

import vulkan;
import std;

namespace engine {

// ---------------------------------------------------------------------------: Vertex

// Vertex layout for glTF PBR. Matches the VertexInput struct in pbr.slang.
// Tangent.w carries the bitangent sign (glTF convention), so the fragment
// shader reconstructs B = cross(N, T.xyz) * T.w for normal mapping.
export struct ModelVertex {
  glm::vec3 position;
  glm::vec3 normal;
  glm::vec2 uv;
  glm::vec4 tangent;

  static vk::VertexInputBindingDescription GetBindingDescription() {
    return vk::VertexInputBindingDescription{
        .binding = 0,
        .stride = sizeof(ModelVertex),
        .inputRate = vk::VertexInputRate::eVertex,
    };
  }

  static std::array<vk::VertexInputAttributeDescription, 4> GetAttributeDescriptions() {
    return {
        vk::VertexInputAttributeDescription{
            .location = 0, .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(ModelVertex, position),
        },
        vk::VertexInputAttributeDescription{
            .location = 1, .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = offsetof(ModelVertex, normal),
        },
        vk::VertexInputAttributeDescription{
            .location = 2, .binding = 0,
            .format = vk::Format::eR32G32Sfloat,
            .offset = offsetof(ModelVertex, uv),
        },
        vk::VertexInputAttributeDescription{
            .location = 3, .binding = 0,
            .format = vk::Format::eR32G32B32A32Sfloat,
            .offset = offsetof(ModelVertex, tangent),
        },
    };
  }
};

// ---------------------------------------------------------------------------: Material

// PBR metallic-roughness material. Texture indices are into Model::textures;
// -1 means "no texture — fall back to the constant factor".
export struct Material {
  glm::vec4 base_color_factor{1.0f};
  glm::vec3 emissive_factor{0.0f};
  float metallic_factor = 1.0f;
  float roughness_factor = 1.0f;
  int base_color_texture = -1;
  int metallic_roughness_texture = -1;
  int normal_texture = -1;
  int occlusion_texture = -1;
  int emissive_texture = -1;
};

// ---------------------------------------------------------------------------: Texture

// Vulkan-side handle for a single loaded image. The sampler is allocated
// per-texture for now; in a larger engine we'd share samplers keyed on the
// glTF sampler definition to cut descriptor pool pressure.
export struct ModelTexture {
  vk::raii::Image image = nullptr;
  vk::raii::DeviceMemory memory = nullptr;
  vk::raii::ImageView view = nullptr;
  vk::raii::Sampler sampler = nullptr;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t mip_levels = 1;
};

// ---------------------------------------------------------------------------: MeshPrimitive

// One drawable chunk: a vertex/index buffer pair plus the material that shades
// it. A glTF mesh can have multiple primitives (e.g. car body + windshield
// share vertex positions but use different materials), so a Node owns a vector
// of these instead of a single mesh.
export struct MeshPrimitive {
  vk::raii::Buffer vertex_buffer = nullptr;
  vk::raii::DeviceMemory vertex_memory = nullptr;
  vk::raii::Buffer index_buffer = nullptr;
  vk::raii::DeviceMemory index_memory = nullptr;
  std::uint32_t index_count = 0;
  int material_index = -1;
};

// ---------------------------------------------------------------------------: Node

// Pointer-based scene graph node. Raw child/parent pointers (non-owning) are
// fine because Model owns every node via unique_ptr and outlives the graph.
//
// glTF lets a node specify either a baked local matrix or TRS components; if
// both are provided (rare), TRS wins per spec. We store both and multiply
// them together in GetLocalMatrix so whichever the loader fills in works.
export struct Node {
  Node* parent = nullptr;
  std::vector<Node*> children;
  std::string name;
  std::uint32_t index = 0;
  std::vector<MeshPrimitive> primitives;

  glm::vec3 translation{0.0f};
  glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec3 scale{1.0f};
  glm::mat4 matrix{1.0f};

  glm::mat4 GetLocalMatrix() const {
    return glm::translate(glm::mat4(1.0f), translation)
         * glm::mat4_cast(rotation)
         * glm::scale(glm::mat4(1.0f), scale)
         * matrix;
  }

  glm::mat4 GetGlobalMatrix() const {
    glm::mat4 m = GetLocalMatrix();
    const Node* p = parent;
    while (p != nullptr) {
      m = p->GetLocalMatrix() * m;
      p = p->parent;
    }
    return m;
  }
};

// ---------------------------------------------------------------------------: Animation

// AnimationSampler::outputs is always stored as vec4 to keep one container for
// every channel type. TRANSLATION/SCALE use xyz and ignore w; ROTATION packs a
// quaternion as (x, y, z, w) matching glTF's on-disk layout.
export struct AnimationSampler {
  enum class Interpolation : std::uint8_t { Linear, Step, CubicSpline };
  Interpolation interpolation = Interpolation::Linear;
  std::vector<float> inputs;
  std::vector<glm::vec4> outputs;
};

export struct AnimationChannel {
  enum class Path : std::uint8_t { Translation, Rotation, Scale };
  Path path = Path::Translation;
  Node* node = nullptr;
  std::uint32_t sampler_index = 0;
};

export struct Animation {
  std::string name;
  std::vector<AnimationSampler> samplers;
  std::vector<AnimationChannel> channels;
  float start = std::numeric_limits<float>::max();
  float end = std::numeric_limits<float>::lowest();
  float current_time = 0.0f;
};

// ---------------------------------------------------------------------------: Model

// Owning container for everything the glTF loader produces. Rendering code
// walks `root_nodes` depth-first and draws every primitive along the way.
export struct Model {
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<Node*> root_nodes;
  std::vector<Material> materials;
  std::vector<ModelTexture> textures;
  std::vector<Animation> animations;

  Node* FindNode(std::string_view name) const {
    for (const auto& node : nodes) {
      if (node->name == name) {
        return node.get();
      }
    }
    return nullptr;
  }

  // Advances the animation clock and writes interpolated TRS back into each
  // channel's target node. Loops automatically — the next frame picks up from
  // the new wrapped-around currentTime.
  void UpdateAnimation(std::uint32_t index, float delta_time) {
    if (animations.empty() || index >= animations.size()) {
      return;
    }
    Animation& animation = animations[index];
    animation.current_time += delta_time;
    const float duration = animation.end - animation.start;
    if (duration > 0.0f) {
      while (animation.current_time > animation.end) {
        animation.current_time -= duration;
      }
    }

    for (auto& channel : animation.channels) {
      if (channel.sampler_index >= animation.samplers.size() || channel.node == nullptr) {
        continue;
      }
      const AnimationSampler& sampler = animation.samplers[channel.sampler_index];
      if (sampler.inputs.size() < 2 || sampler.outputs.size() < 2) {
        continue;
      }

      // Find keyframe pair bracketing current_time via binary search.
      auto next = std::ranges::lower_bound(sampler.inputs, animation.current_time);
      if (next == sampler.inputs.begin() || next == sampler.inputs.end()) {
        continue;
      }
      const std::size_t i = static_cast<std::size_t>(next - sampler.inputs.begin()) - 1;
      const float t_start = sampler.inputs[i];
      const float t_end = sampler.inputs[i + 1];
      const float span = t_end - t_start;
      const float t = span > 0.0f ? (animation.current_time - t_start) / span : 0.0f;

      switch (channel.path) {
        case AnimationChannel::Path::Translation: {
          const glm::vec3 a(sampler.outputs[i]);
          const glm::vec3 b(sampler.outputs[i + 1]);
          channel.node->translation = glm::mix(a, b, t);
          break;
        }
        case AnimationChannel::Path::Scale: {
          const glm::vec3 a(sampler.outputs[i]);
          const glm::vec3 b(sampler.outputs[i + 1]);
          channel.node->scale = glm::mix(a, b, t);
          break;
        }
        case AnimationChannel::Path::Rotation: {
          const glm::vec4& va = sampler.outputs[i];
          const glm::vec4& vb = sampler.outputs[i + 1];
          const glm::quat qa(va.w, va.x, va.y, va.z);
          const glm::quat qb(vb.w, vb.x, vb.y, vb.z);
          channel.node->rotation = glm::normalize(glm::slerp(qa, qb, t));
          break;
        }
      }
    }
  }
};

}  // namespace engine
