module;

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>

export module engine.physics;

import std;
import vulkan;
import engine.device;
import engine.scene;

namespace engine {

// ---------------------------------------------------------------------------: Collider

export enum class ColliderType { Box, Sphere };

// Base collider. SetOffset lets users nudge the collider relative to the body
// origin (e.g. a sphere whose center is above the body root). The concrete
// shapes live below and expose shape-specific parameters.
export class Collider {
 public:
  virtual ~Collider() = default;
  virtual ColliderType GetType() const = 0;

  void SetOffset(const glm::vec3& offset) { offset_ = offset; }
  const glm::vec3& GetOffset() const { return offset_; }

 protected:
  glm::vec3 offset_{0.0f};
};

export class SphereCollider : public Collider {
 public:
  explicit SphereCollider(float radius) : radius_(radius) {}
  ColliderType GetType() const override { return ColliderType::Sphere; }
  float GetRadius() const { return radius_; }
  void SetRadius(float r) { radius_ = r; }

 private:
  float radius_ = 0.5f;
};

// Axis-aligned box collider — body rotation is ignored on BoxColliders in ep8.
// Good enough for static ground planes and axis-aligned walls; OBB support is
// a follow-up (requires SAT in collision detection).
export class BoxCollider : public Collider {
 public:
  explicit BoxCollider(const glm::vec3& half_extents) : half_extents_(half_extents) {}
  ColliderType GetType() const override { return ColliderType::Box; }
  const glm::vec3& GetHalfExtents() const { return half_extents_; }
  void SetHalfExtents(const glm::vec3& he) { half_extents_ = he; }

 private:
  glm::vec3 half_extents_{0.5f};
};

// ---------------------------------------------------------------------------: RigidBody

// Rigid body state + forces. Position/rotation are authoritative during physics
// steps; the RigidBodyComponent pushes them onto the owning entity's
// TransformComponent each frame. Angular velocity is integrated (so
// ApplyTorqueImpulse produces visible spin) but collision resolution is
// linear-only — the tutorial does the same and we match.
export class RigidBody {
 public:
  RigidBody() { RebuildInertia(); }

  // --- pose + motion ------------------------------------------------------
  void SetPosition(const glm::vec3& p) { position_ = p; }
  void SetRotation(const glm::quat& r) { rotation_ = r; }
  void SetLinearVelocity(const glm::vec3& v) { linear_velocity_ = v; }
  void SetAngularVelocity(const glm::vec3& v) { angular_velocity_ = v; }

  const glm::vec3& GetPosition() const { return position_; }
  const glm::quat& GetRotation() const { return rotation_; }
  const glm::vec3& GetLinearVelocity() const { return linear_velocity_; }
  const glm::vec3& GetAngularVelocity() const { return angular_velocity_; }

  // --- material properties ------------------------------------------------
  void SetMass(float mass) {
    mass_ = mass;
    inverse_mass_ = mass > 0.0f ? 1.0f / mass : 0.0f;
    RebuildInertia();
  }
  float GetMass() const { return mass_; }
  float GetInverseMass() const { return inverse_mass_; }
  const glm::mat3& GetInverseInertia() const { return inverse_inertia_; }

  void SetRestitution(float e) { restitution_ = e; }
  float GetRestitution() const { return restitution_; }
  void SetFriction(float f) { friction_ = f; }
  float GetFriction() const { return friction_; }

  // --- collider -----------------------------------------------------------
  void SetCollider(std::unique_ptr<Collider> collider) {
    collider_ = std::move(collider);
    RebuildInertia();
  }
  Collider* GetCollider() const { return collider_.get(); }

  // --- forces -------------------------------------------------------------
  void ApplyForce(const glm::vec3& f) { accumulated_force_ += f; }
  void ApplyImpulse(const glm::vec3& j) {
    if (!is_kinematic_) {
      linear_velocity_ += j * inverse_mass_;
    }
  }
  void ApplyTorque(const glm::vec3& t) { accumulated_torque_ += t; }
  void ApplyTorqueImpulse(const glm::vec3& ti) {
    if (!is_kinematic_) {
      angular_velocity_ += inverse_inertia_ * ti;
    }
  }

  // --- flags --------------------------------------------------------------
  void SetKinematic(bool k) { is_kinematic_ = k; }
  bool IsKinematic() const { return is_kinematic_; }
  void SetGravityEnabled(bool g) { use_gravity_ = g; }
  bool IsGravityEnabled() const { return use_gravity_; }

 private:
  friend class PhysicsSystem;

  // Crude diagonal inertia tensor from current mass + collider. Sphere:
  // I = (2/5) m r². Box: I = (m/3)(he_y² + he_z², he_x² + he_z², he_x² + he_y²).
  // Good enough for a toy physics system; a proper engine would let users
  // provide a custom tensor.
  void RebuildInertia() {
    inverse_inertia_ = glm::mat3(0.0f);
    if (mass_ <= 0.0f || collider_ == nullptr) {
      return;
    }
    glm::mat3 inertia(0.0f);
    if (collider_->GetType() == ColliderType::Sphere) {
      auto* sphere = static_cast<SphereCollider*>(collider_.get());
      float i = 0.4f * mass_ * sphere->GetRadius() * sphere->GetRadius();
      inertia = glm::mat3(i);
    } else {
      auto* box = static_cast<BoxCollider*>(collider_.get());
      const glm::vec3 he = box->GetHalfExtents();
      float s = mass_ / 3.0f;
      inertia[0][0] = s * (he.y * he.y + he.z * he.z);
      inertia[1][1] = s * (he.x * he.x + he.z * he.z);
      inertia[2][2] = s * (he.x * he.x + he.y * he.y);
    }
    inverse_inertia_ = glm::inverse(inertia);
  }

  glm::vec3 position_{0.0f};
  glm::quat rotation_{1.0f, 0.0f, 0.0f, 0.0f};
  glm::vec3 linear_velocity_{0.0f};
  glm::vec3 angular_velocity_{0.0f};

  glm::vec3 accumulated_force_{0.0f};
  glm::vec3 accumulated_torque_{0.0f};

  float mass_ = 1.0f;
  float inverse_mass_ = 1.0f;
  glm::mat3 inverse_inertia_{0.0f};
  float restitution_ = 0.3f;
  float friction_ = 0.5f;

  std::unique_ptr<Collider> collider_;
  bool is_kinematic_ = false;
  bool use_gravity_ = true;
};

// ---------------------------------------------------------------------------: CollisionInfo

// Populated by PhysicsSystem's per-pair CheckCollision. body_a / body_b are
// non-owning — they point into PhysicsSystem::bodies_. Normal points from A
// toward B (so A gets pushed opposite, B along normal).
export struct CollisionInfo {
  RigidBody* body_a = nullptr;
  RigidBody* body_b = nullptr;
  glm::vec3 contact_point{0.0f};
  glm::vec3 normal{0.0f, 1.0f, 0.0f};
  float penetration_depth = 0.0f;
};

// ---------------------------------------------------------------------------: GpuPhysicsData

// Flat std430-compatible body state for the compute shader. Layout must match
// `struct PhysicsData` in assets/shaders/physics.slang — 128 bytes per body,
// eight vec4 slots. Scalars ride in the w channels so the struct stays
// fully vec4-aligned:
//   position.w          = inverse mass
//   linear_velocity.w   = restitution
//   angular_velocity.w  = friction (reserved — unused by stepMain today)
//   force.w             = is_kinematic flag (0/1)
//   torque.w            = use_gravity flag (0/1)
//   collider.w          = collider type sentinel (1 = sphere, 0 = box, -1 = none)
//   collider.xyz        = sphere radius or box half-extents
//   collider2.xyz       = collider offset from body origin
export struct GpuPhysicsData {
  glm::vec4 position;
  glm::vec4 rotation;
  glm::vec4 linear_velocity;
  glm::vec4 angular_velocity;
  glm::vec4 force;
  glm::vec4 torque;
  glm::vec4 collider;
  glm::vec4 collider2;
};
static_assert(sizeof(GpuPhysicsData) == 128,
              "GpuPhysicsData must match shader-side std430 layout (8 * vec4)");

// ---------------------------------------------------------------------------: GpuPhysicsProcessor

// Compute-backed rigid-body stepper. Uploads a flat array of GpuPhysicsData
// into a host-visible storage buffer, dispatches physics.slang's stepMain,
// waits on a fence, reads the updated state back. The storage buffer grows on
// demand — small body counts don't pay for a million-body allocation up front.
//
// Reuses the graphics queue (same family chosen in Device::FindGraphicsQueueFamily
// already requires COMPUTE) so this demo stays on one queue; a dedicated
// async-compute queue is a later concern for real-time physics overlap with
// rendering.
export class GpuPhysicsProcessor {
 public:
  explicit GpuPhysicsProcessor(Device& device) : device_(device) {
    const auto& dev = device_.GetLogicalDevice();

    vk::DescriptorSetLayoutBinding binding{
        .binding = 0,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
    };
    vk::DescriptorSetLayoutCreateInfo set_layout_info{
        .bindingCount = 1,
        .pBindings = &binding,
    };
    set_layout_ = vk::raii::DescriptorSetLayout(dev, set_layout_info);

    vk::PushConstantRange push_range{
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(PushConstants),
    };
    vk::PipelineLayoutCreateInfo layout_info{
        .setLayoutCount = 1,
        .pSetLayouts = &*set_layout_,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_range,
    };
    pipe_layout_ = vk::raii::PipelineLayout(dev, layout_info);

    std::ifstream file(kShaderPath, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error(std::string("GpuPhysicsProcessor: failed to open ") + kShaderPath);
    }
    std::vector<char> code(static_cast<std::size_t>(file.tellg()));
    file.seekg(0);
    file.read(code.data(), static_cast<std::streamsize>(code.size()));
    vk::ShaderModuleCreateInfo shader_info{
        .codeSize = code.size(),
        .pCode = reinterpret_cast<const std::uint32_t*>(code.data()),
    };
    shader_ = vk::raii::ShaderModule(dev, shader_info);

    vk::PipelineShaderStageCreateInfo stage{
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *shader_,
        .pName = "stepMain",
    };
    vk::ComputePipelineCreateInfo pipe_info{
        .stage = stage,
        .layout = *pipe_layout_,
    };
    pipeline_ = vk::raii::Pipeline(dev, nullptr, pipe_info);

    vk::DescriptorPoolSize pool_size{
        .type = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 1,
    };
    vk::DescriptorPoolCreateInfo pool_info{
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
    };
    pool_ = vk::raii::DescriptorPool(dev, pool_info);

    vk::DescriptorSetAllocateInfo set_alloc_info{
        .descriptorPool = *pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = &*set_layout_,
    };
    auto sets = vk::raii::DescriptorSets(dev, set_alloc_info);
    set_ = std::move(sets[0]);

    vk::CommandPoolCreateInfo cmd_pool_info{
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = device_.GetGraphicsQueueFamily(),
    };
    cmd_pool_ = vk::raii::CommandPool(dev, cmd_pool_info);

    vk::CommandBufferAllocateInfo cmd_alloc_info{
        .commandPool = *cmd_pool_,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };
    auto cmds = vk::raii::CommandBuffers(dev, cmd_alloc_info);
    cmd_ = std::move(cmds[0]);

    fence_ = vk::raii::Fence(dev, vk::FenceCreateInfo{});
    // bodies_buf_ is allocated lazily in EnsureCapacity so we don't size for a
    // worst-case body count that may never materialize.
  }

  GpuPhysicsProcessor(const GpuPhysicsProcessor&) = delete;
  GpuPhysicsProcessor& operator=(const GpuPhysicsProcessor&) = delete;
  GpuPhysicsProcessor(GpuPhysicsProcessor&&) = delete;
  GpuPhysicsProcessor& operator=(GpuPhysicsProcessor&&) = delete;

  // In-place: uploads `bodies`, dispatches one thread per body, reads the
  // updated state back into the same span. `dt` is a fixed sim step (not
  // wall-clock frame time) and `ground_y` is the world-space height of the
  // single implicit ground plane the shader resolves against.
  void Step(std::span<GpuPhysicsData> bodies, float dt,
            const glm::vec3& gravity, float ground_y) {
    if (bodies.empty()) {
      return;
    }
    EnsureCapacity(bodies.size());

    std::memcpy(bodies_buf_.mapped, bodies.data(), bodies.size_bytes());

    const auto& dev = device_.GetLogicalDevice();

    cmd_.reset();
    vk::CommandBufferBeginInfo begin_info{
        .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit,
    };
    cmd_.begin(begin_info);
    cmd_.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline_);
    cmd_.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipe_layout_, 0,
                            *set_, nullptr);

    PushConstants pc{
        .dt = dt,
        .ground_y = ground_y,
        .body_count = static_cast<std::uint32_t>(bodies.size()),
        ._pad0 = 0.0f,
        .gravity = glm::vec4(gravity, 0.0f),
    };
    cmd_.pushConstants<PushConstants>(*pipe_layout_, vk::ShaderStageFlagBits::eCompute,
                                      0, pc);

    const std::uint32_t group_count = (pc.body_count + 63) / 64;
    cmd_.dispatch(group_count, 1, 1);

    // Host-coherent memory makes fence signaling sufficient in practice, but
    // the explicit shader-write → host-read barrier keeps the validation layer
    // quiet and documents the dependency — same pattern as HrtfProcessor.
    vk::MemoryBarrier mem_barrier{
        .srcAccessMask = vk::AccessFlagBits::eShaderWrite,
        .dstAccessMask = vk::AccessFlagBits::eHostRead,
    };
    cmd_.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                         vk::PipelineStageFlagBits::eHost,
                         {}, mem_barrier, nullptr, nullptr);
    cmd_.end();

    dev.resetFences(*fence_);
    vk::SubmitInfo submit_info{
        .commandBufferCount = 1,
        .pCommandBuffers = &*cmd_,
    };
    device_.GetGraphicsQueue().submit(submit_info, *fence_);

    auto wait_result = dev.waitForFences(*fence_, vk::True,
                                         std::numeric_limits<std::uint64_t>::max());
    if (wait_result != vk::Result::eSuccess) {
      throw std::runtime_error("GpuPhysicsProcessor: fence wait failed");
    }

    std::memcpy(bodies.data(), bodies_buf_.mapped, bodies.size_bytes());
  }

 private:
  // Matches `struct Params` in assets/shaders/physics.slang — 32 bytes.
  struct PushConstants {
    float dt;
    float ground_y;
    std::uint32_t body_count;
    float _pad0;
    glm::vec4 gravity;
  };
  static_assert(sizeof(PushConstants) == 32);

  struct HostBuffer {
    vk::raii::Buffer buf = nullptr;
    vk::raii::DeviceMemory mem = nullptr;
    void* mapped = nullptr;
    vk::DeviceSize size = 0;
  };

  void EnsureCapacity(std::size_t body_count) {
    const vk::DeviceSize bytes = body_count * sizeof(GpuPhysicsData);
    if (bodies_buf_.size >= bytes) {
      return;
    }
    bodies_buf_ = HostBuffer{};
    CreateHostBuffer(bodies_buf_, bytes);
    WriteDescriptorSet();
  }

  void CreateHostBuffer(HostBuffer& dst, vk::DeviceSize size) {
    const auto& dev = device_.GetLogicalDevice();
    vk::BufferCreateInfo buf_info{
        .size = size,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
    };
    dst.buf = vk::raii::Buffer(dev, buf_info);
    vk::MemoryRequirements reqs = dst.buf.getMemoryRequirements();
    vk::MemoryAllocateInfo alloc_info{
        .allocationSize = reqs.size,
        .memoryTypeIndex = FindMemoryType(
            reqs.memoryTypeBits,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
    };
    dst.mem = vk::raii::DeviceMemory(dev, alloc_info);
    dst.buf.bindMemory(*dst.mem, 0);
    dst.mapped = dst.mem.mapMemory(0, size);
    dst.size = size;
  }

  void WriteDescriptorSet() {
    vk::DescriptorBufferInfo buffer_info{
        .buffer = *bodies_buf_.buf,
        .offset = 0,
        .range = bodies_buf_.size,
    };
    vk::WriteDescriptorSet write{
        .dstSet = *set_,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eStorageBuffer,
        .pBufferInfo = &buffer_info,
    };
    device_.GetLogicalDevice().updateDescriptorSets(write, nullptr);
  }

  std::uint32_t FindMemoryType(std::uint32_t type_filter,
                               vk::MemoryPropertyFlags properties) const {
    auto props = device_.GetPhysicalDevice().getMemoryProperties();
    for (std::uint32_t i = 0; i < props.memoryTypeCount; ++i) {
      bool type_ok = (type_filter & (1u << i)) != 0;
      bool props_ok = (props.memoryTypes[i].propertyFlags & properties) == properties;
      if (type_ok && props_ok) {
        return i;
      }
    }
    throw std::runtime_error("GpuPhysicsProcessor: no suitable memory type");
  }

  static constexpr const char* kShaderPath = "assets/shaders/physics.spv";

  Device& device_;

  vk::raii::DescriptorSetLayout set_layout_ = nullptr;
  vk::raii::PipelineLayout pipe_layout_ = nullptr;
  vk::raii::ShaderModule shader_ = nullptr;
  vk::raii::Pipeline pipeline_ = nullptr;
  vk::raii::DescriptorPool pool_ = nullptr;
  vk::raii::DescriptorSet set_ = nullptr;
  vk::raii::CommandPool cmd_pool_ = nullptr;
  vk::raii::CommandBuffer cmd_ = nullptr;
  vk::raii::Fence fence_ = nullptr;

  HostBuffer bodies_buf_;
};

// ---------------------------------------------------------------------------: PhysicsSystem

// CPU rigid-body simulation. One step: accumulate forces → integrate forces →
// detect collisions (naive O(n²)) → resolve collisions (impulse + positional
// correction, linear-only) → integrate velocities → clear forces. Outer
// Update() drives fixed 1/60 s steps via an accumulator so sim is framerate-
// independent.
export class PhysicsSystem {
 public:
  PhysicsSystem() = default;
  ~PhysicsSystem() = default;

  PhysicsSystem(const PhysicsSystem&) = delete;
  PhysicsSystem& operator=(const PhysicsSystem&) = delete;
  PhysicsSystem(PhysicsSystem&&) = delete;
  PhysicsSystem& operator=(PhysicsSystem&&) = delete;

  // Returns a non-owning pointer. Caller attaches a collider via body->SetCollider.
  RigidBody* CreateRigidBody() {
    bodies_.push_back(std::make_unique<RigidBody>());
    return bodies_.back().get();
  }

  // Removes a body by address. Linear scan — body counts stay small in demos.
  void DestroyRigidBody(RigidBody* body) {
    auto it = std::ranges::find_if(
        bodies_, [body](const auto& up) { return up.get() == body; });
    if (it != bodies_.end()) {
      bodies_.erase(it);
    }
  }

  void SetGravity(const glm::vec3& g) { gravity_ = g; }
  const glm::vec3& GetGravity() const { return gravity_; }

  std::size_t GetBodyCount() const { return bodies_.size(); }

  // Main loop hook. Caps the accumulated delta at 0.25 s so breakpoints or
  // long stalls don't unwind into hundreds of fixed steps at once.
  void Update(float delta_time) {
    constexpr float kFixedStep = 1.0f / 60.0f;
    constexpr float kMaxAccum = 0.25f;
    accumulator_ += std::min(delta_time, kMaxAccum);
    while (accumulator_ >= kFixedStep) {
      Step(kFixedStep);
      accumulator_ -= kFixedStep;
    }
  }

  // Alternate loop hook: runs each fixed step through the compute-shader
  // pipeline in GpuPhysicsProcessor. Only collision handled is sphere vs. a
  // single horizontal ground plane at `ground_y` — sphere-vs-sphere contact is
  // not done on the GPU in this first pass (see physics.slang). Accumulator is
  // shared with Update(), so toggling between CPU and GPU mid-simulation
  // doesn't drop or double-count pending time.
  void UpdateGpu(GpuPhysicsProcessor& processor, float delta_time, float ground_y) {
    constexpr float kFixedStep = 1.0f / 60.0f;
    constexpr float kMaxAccum = 0.25f;
    accumulator_ += std::min(delta_time, kMaxAccum);
    while (accumulator_ >= kFixedStep) {
      StepGpu(processor, kFixedStep, ground_y);
      accumulator_ -= kFixedStep;
    }
  }

 private:
  void Step(float dt) {
    AccumulateForces();
    IntegrateForces(dt);
    std::vector<CollisionInfo> collisions;
    DetectCollisions(collisions);
    ResolveCollisions(collisions);
    IntegrateVelocities(dt);
    ClearForces();
  }

  // One GPU-driven fixed step: pack state → dispatch stepMain → unpack state.
  // Kinematic bodies are packed (the shader needs to see them for counts and
  // flags) but are early-outed inside the shader. Kept in the same fixed-step
  // wrapper as Step() so toggling CPU/GPU doesn't change the time integrator.
  void StepGpu(GpuPhysicsProcessor& processor, float dt, float ground_y) {
    if (bodies_.empty()) {
      return;
    }
    gpu_scratch_.resize(bodies_.size());
    for (std::size_t i = 0; i < bodies_.size(); ++i) {
      gpu_scratch_[i] = PackBody(*bodies_[i]);
    }
    processor.Step(std::span(gpu_scratch_), dt, gravity_, ground_y);
    for (std::size_t i = 0; i < bodies_.size(); ++i) {
      UnpackBody(*bodies_[i], gpu_scratch_[i]);
    }
  }

  static GpuPhysicsData PackBody(const RigidBody& body) {
    GpuPhysicsData out{};
    out.position = glm::vec4(body.position_, body.inverse_mass_);
    const glm::quat& q = body.rotation_;
    out.rotation = glm::vec4(q.x, q.y, q.z, q.w);
    out.linear_velocity = glm::vec4(body.linear_velocity_, body.restitution_);
    out.angular_velocity = glm::vec4(body.angular_velocity_, body.friction_);
    out.force = glm::vec4(body.accumulated_force_, body.is_kinematic_ ? 1.0f : 0.0f);
    out.torque = glm::vec4(body.accumulated_torque_, body.use_gravity_ ? 1.0f : 0.0f);

    if (body.collider_ == nullptr) {
      out.collider = glm::vec4(0.0f, 0.0f, 0.0f, -1.0f);
      out.collider2 = glm::vec4(0.0f);
      return out;
    }
    if (body.collider_->GetType() == ColliderType::Sphere) {
      const auto* s = static_cast<const SphereCollider*>(body.collider_.get());
      out.collider = glm::vec4(s->GetRadius(), 0.0f, 0.0f, 1.0f);
    } else {
      const auto* b = static_cast<const BoxCollider*>(body.collider_.get());
      out.collider = glm::vec4(b->GetHalfExtents(), 0.0f);
    }
    out.collider2 = glm::vec4(body.collider_->GetOffset(), 0.0f);
    return out;
  }

  static void UnpackBody(RigidBody& body, const GpuPhysicsData& gpu) {
    if (body.is_kinematic_) {
      return;  // shader early-outs kinematic bodies, so GPU state is stale
    }
    body.position_ = glm::vec3(gpu.position);
    body.rotation_ = glm::quat(gpu.rotation.w, gpu.rotation.x, gpu.rotation.y, gpu.rotation.z);
    body.linear_velocity_ = glm::vec3(gpu.linear_velocity);
    body.angular_velocity_ = glm::vec3(gpu.angular_velocity);
    // Shader already cleared force/torque into the w-flag-preserving slots;
    // mirror that on CPU so any ApplyForce next frame starts from zero.
    body.accumulated_force_ = glm::vec3(0.0f);
    body.accumulated_torque_ = glm::vec3(0.0f);
  }

  void AccumulateForces() {
    for (auto& body : bodies_) {
      if (body->is_kinematic_ || !body->use_gravity_) {
        continue;
      }
      body->accumulated_force_ += gravity_ * body->mass_;
    }
  }

  void IntegrateForces(float dt) {
    constexpr float kLinearDamping = 0.01f;
    constexpr float kAngularDamping = 0.01f;
    for (auto& body : bodies_) {
      if (body->is_kinematic_) {
        continue;
      }
      body->linear_velocity_ += body->accumulated_force_ * body->inverse_mass_ * dt;
      body->angular_velocity_ += body->inverse_inertia_ * body->accumulated_torque_ * dt;
      body->linear_velocity_ *= (1.0f - kLinearDamping);
      body->angular_velocity_ *= (1.0f - kAngularDamping);
    }
  }

  void DetectCollisions(std::vector<CollisionInfo>& out) {
    for (std::size_t i = 0; i < bodies_.size(); ++i) {
      for (std::size_t j = i + 1; j < bodies_.size(); ++j) {
        RigidBody& a = *bodies_[i];
        RigidBody& b = *bodies_[j];
        if (a.is_kinematic_ && b.is_kinematic_) {
          continue;
        }
        if (a.collider_ == nullptr || b.collider_ == nullptr) {
          continue;
        }
        CollisionInfo info;
        if (CheckCollision(a, b, info)) {
          info.body_a = &a;
          info.body_b = &b;
          out.push_back(info);
        }
      }
    }
  }

  void ResolveCollisions(const std::vector<CollisionInfo>& collisions) {
    for (const auto& c : collisions) {
      RigidBody& a = *c.body_a;
      RigidBody& b = *c.body_b;
      float inv_a = a.is_kinematic_ ? 0.0f : a.inverse_mass_;
      float inv_b = b.is_kinematic_ ? 0.0f : b.inverse_mass_;
      float inv_sum = inv_a + inv_b;
      if (inv_sum <= 0.0f) {
        continue;
      }

      glm::vec3 rel_vel = b.linear_velocity_ - a.linear_velocity_;
      float vel_along_normal = glm::dot(rel_vel, c.normal);
      if (vel_along_normal > 0.0f) {
        continue;  // separating — no impulse
      }

      float e = std::min(a.restitution_, b.restitution_);
      // Resting-contact guard: gravity injects ~|g|·dt of approach speed each
      // step (≈0.163 m/s at 60 Hz). Without zeroing restitution at low speeds,
      // that speed gets bounced back up and the body vibrates on the ground.
      constexpr float kRestSpeedThreshold = 1.0f;
      if (-vel_along_normal < kRestSpeedThreshold) {
        e = 0.0f;
      }
      float j = -(1.0f + e) * vel_along_normal / inv_sum;
      glm::vec3 impulse = c.normal * j;
      if (!a.is_kinematic_) {
        a.linear_velocity_ -= impulse * inv_a;
      }
      if (!b.is_kinematic_) {
        b.linear_velocity_ += impulse * inv_b;
      }

      // Positional correction: pushes bodies out of penetration so they don't
      // sink through each other. `slop` avoids jitter at rest.
      constexpr float kCorrectPercent = 0.2f;
      constexpr float kSlop = 0.01f;
      float pen = std::max(c.penetration_depth - kSlop, 0.0f);
      glm::vec3 correction = (pen / inv_sum) * kCorrectPercent * c.normal;
      if (!a.is_kinematic_) {
        a.position_ -= correction * inv_a;
      }
      if (!b.is_kinematic_) {
        b.position_ += correction * inv_b;
      }
    }
  }

  void IntegrateVelocities(float dt) {
    for (auto& body : bodies_) {
      if (body->is_kinematic_) {
        continue;
      }
      body->position_ += body->linear_velocity_ * dt;
      // Quaternion derivative q̇ = ½ ω q. Normalize each step to keep drift
      // bounded (tutorial pattern).
      glm::quat omega_q(0.0f, body->angular_velocity_.x, body->angular_velocity_.y,
                        body->angular_velocity_.z);
      body->rotation_ += (omega_q * body->rotation_) * (0.5f * dt);
      body->rotation_ = glm::normalize(body->rotation_);
    }
  }

  void ClearForces() {
    for (auto& body : bodies_) {
      body->accumulated_force_ = glm::vec3(0.0f);
      body->accumulated_torque_ = glm::vec3(0.0f);
    }
  }

  bool CheckCollision(RigidBody& a, RigidBody& b, CollisionInfo& out) {
    auto* ca = a.collider_.get();
    auto* cb = b.collider_.get();
    if (ca->GetType() == ColliderType::Sphere && cb->GetType() == ColliderType::Sphere) {
      return SphereVsSphere(a, b, out);
    }
    if (ca->GetType() == ColliderType::Sphere && cb->GetType() == ColliderType::Box) {
      return SphereVsAabb(a, b, out);
    }
    if (ca->GetType() == ColliderType::Box && cb->GetType() == ColliderType::Sphere) {
      bool hit = SphereVsAabb(b, a, out);
      if (hit) {
        // Normal was computed from sphere toward box center; our contract is
        // normal goes a→b, so flip since we swapped roles.
        out.normal = -out.normal;
      }
      return hit;
    }
    // Box×Box not implemented — scoped out of ep8.
    return false;
  }

  bool SphereVsSphere(RigidBody& a, RigidBody& b, CollisionInfo& out) {
    auto* sa = static_cast<SphereCollider*>(a.collider_.get());
    auto* sb = static_cast<SphereCollider*>(b.collider_.get());
    glm::vec3 pa = a.position_ + sa->GetOffset();
    glm::vec3 pb = b.position_ + sb->GetOffset();
    float ra = sa->GetRadius();
    float rb = sb->GetRadius();
    glm::vec3 diff = pb - pa;
    float dist_sq = glm::dot(diff, diff);
    float min_dist = ra + rb;
    if (dist_sq >= min_dist * min_dist) {
      return false;
    }
    float dist = std::sqrt(dist_sq);
    glm::vec3 normal = dist > 1e-5f ? diff / dist : glm::vec3(0.0f, 1.0f, 0.0f);
    out.normal = normal;
    out.contact_point = pa + normal * ra;
    out.penetration_depth = min_dist - dist;
    return true;
  }

  // Sphere at `sphere` vs AABB at `box`. Assumes box rotation is identity —
  // documented as a BoxCollider limitation above.
  bool SphereVsAabb(RigidBody& sphere, RigidBody& box, CollisionInfo& out) {
    auto* s = static_cast<SphereCollider*>(sphere.collider_.get());
    auto* b = static_cast<BoxCollider*>(box.collider_.get());
    glm::vec3 sp = sphere.position_ + s->GetOffset();
    glm::vec3 bp = box.position_ + b->GetOffset();
    glm::vec3 he = b->GetHalfExtents();
    float radius = s->GetRadius();

    glm::vec3 local = sp - bp;
    glm::vec3 clamped = glm::clamp(local, -he, he);
    glm::vec3 diff = local - clamped;
    float dist_sq = glm::dot(diff, diff);
    if (dist_sq >= radius * radius) {
      return false;
    }

    if (dist_sq > 1e-6f) {
      float dist = std::sqrt(dist_sq);
      // `diff` points box → sphere; CollisionInfo contract wants A→B
      // (sphere → box here), so negate. SphereVsSphere already uses A→B via
      // `pb - pa`; without this flip SphereVsAabb is inconsistent and the
      // approach-test in ResolveCollisions fires on the wrong side.
      out.normal = -diff / dist;
      out.penetration_depth = radius - dist;
      out.contact_point = bp + clamped;
    } else {
      // Sphere center inside the box — pick the nearest face to push out on.
      glm::vec3 face_dist = he - glm::abs(local);  // +ve = distance to face
      int axis = 0;
      if (face_dist.y < face_dist[axis]) axis = 1;
      if (face_dist.z < face_dist[axis]) axis = 2;
      glm::vec3 n(0.0f);
      // A→B: from sphere center toward box center along the chosen axis.
      n[axis] = local[axis] >= 0.0f ? -1.0f : 1.0f;
      out.normal = n;
      out.penetration_depth = radius + face_dist[axis];
      out.contact_point = sp;
    }
    return true;
  }

  std::vector<std::unique_ptr<RigidBody>> bodies_;
  glm::vec3 gravity_{0.0f, -9.81f, 0.0f};
  float accumulator_ = 0.0f;

  // Reused buffer to avoid per-step allocations on the GPU path. Empty on the
  // CPU path — StepGpu owns its lifetime.
  std::vector<GpuPhysicsData> gpu_scratch_;
};

// ---------------------------------------------------------------------------: RigidBodyComponent

// Bridges a physics body into the scene graph: each frame writes body pose
// onto the owning entity's TransformComponent. Flow is one-way (physics →
// transform); mutate physics via ApplyForce/Impulse on the underlying body,
// never by moving the transform directly.
export class RigidBodyComponent : public Component {
 public:
  explicit RigidBodyComponent(RigidBody* body) : body_(body) {}

  void Update(float /*delta_time*/) override {
    auto* transform = GetOwner()->GetComponent<TransformComponent>();
    if (transform == nullptr || body_ == nullptr) {
      return;
    }
    transform->SetPosition(body_->GetPosition());
    transform->SetRotation(body_->GetRotation());
  }

  RigidBody* Get() const { return body_; }

 private:
  RigidBody* body_ = nullptr;
};

}  // namespace engine
